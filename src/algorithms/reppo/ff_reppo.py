import logging
import math
from typing import Callable
import operator
import os

import hydra
import jax
import optax
import distrax
from flax import nnx
from jax import numpy as jnp
from omegaconf import DictConfig
from gymnax.environments.spaces import Space, Box, Discrete
from src.algorithms.reppo.common import REPPOTrainState
from src.common import (
    InitFn,
    Key,
    LearnerFn,
    Policy,
    Transition,
)
from src.normalization import Normalizer
from src.algorithms import utils

logging.basicConfig(level=logging.INFO)

def batch_norm_phi(
    features: jax.Array,
    scale: jax.Array,
    bias: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """Trainable batch normalization for critic features φ.
    The learnable parameters are the affine scale and bias: φ_bn = scale * ((φ - mean_B) / sqrt(var_B + eps)) + bias.
    """
    mean = features.mean(axis=0, keepdims=True)
    var = jnp.mean(jnp.square(features - mean), axis=0, keepdims=True)
    normalized = (features - mean) / jnp.sqrt(var + eps)
    return normalized * scale + bias

def clip_action_for_critic(action: jax.Array, action_space: Space) -> jax.Array:
    return action.clip(-0.999, 0.999) if isinstance(action_space, Box) else action

def sample_actor_action(actor_model: nnx.Module, obs: jax.Array, key: jax.Array, discrete_actions: bool) -> jax.Array:
    pi = actor_model(obs)
    if discrete_actions:
        return pi.sample(seed=key)
    action, _ = pi.sample_and_log_prob(seed=key)
    return action.clip(-0.999, 0.999)

def gershgorin_loss(gram: jax.Array, feature_dynamics: jax.Array, gamma: float, eps: float = 1e-5):
    """Gershgorin loss for positive stability of A = G(I - γF).

    For TD we need Re(λ(A)) > 0, so we apply that loss to M = -A:
        max(0, -A_ii + Σ_{j≠i}|A_ij| + eps).

    G is computed from stop-gradient features and gradients flow through F.
    """
    gram = jax.lax.stop_gradient(gram)
    feature_dim = gram.shape[-1]
    identity = jnp.eye(feature_dim, dtype=gram.dtype)
    td_matrix = gram @ (identity - gamma * feature_dynamics)

    diag = jnp.diag(td_matrix)
    off_diag_radius = jnp.sum(jnp.abs(td_matrix), axis=-1) - jnp.abs(diag)
    violation = jax.nn.relu(-diag + off_diag_radius + eps)
    loss = jnp.sum(violation)
    margin = diag - off_diag_radius

    return loss, td_matrix, {
        "sr_dice/gershgorin_loss": loss,
    }

def fit_sr_dice_ratio(
    key: jax.Array,
    dice_params: dict[str, jax.Array],
    train_state: REPPOTrainState,
    batch: Transition,
    initial_obs: jax.Array,
    hparams,
    action_space: Space,
    discrete_actions: bool,
):
    """Trainable SR-DICE components on batch-normalized critic features.

    ψ(s,a) = φ_bn(s,a) S, where S is a persistent linear successor-feature layer.
    ρ(s,a) = φ_bn(s,a)^T ν, where ν is a persistent linear ratio head.
    Actor and critic features are fixed for this DICE update; only S and ν are updated.
    """
    key, next_action_key, start_action_key = jax.random.split(key, 3)
    actor_ref = nnx.merge(train_state.actor.graphdef, train_state.actor_target.params)
    critic_ref = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
    actor_ref.eval()
    critic_ref.eval()

    # Accept both full rollout batches [T, N, ...] and already-flattened minibatches [B, ...].
    batch_is_sequence = batch.done.ndim > 1

    obs = batch.obs.reshape((-1, *batch.obs.shape[2:])) if batch_is_sequence else batch.obs
    next_obs = batch.next_obs.reshape((-1, *batch.next_obs.shape[2:])) if batch_is_sequence else batch.next_obs
    behavior_action = batch.action.reshape((-1, *batch.action.shape[2:])) if batch_is_sequence else batch.action
    behavior_action = clip_action_for_critic(behavior_action, action_space)
    done = batch.done.reshape((-1, *batch.done.shape[2:])) if batch_is_sequence else batch.done
    truncated = batch.truncated.reshape((-1, *batch.truncated.shape[2:])) if batch_is_sequence else batch.truncated
    done = done.astype(obs.dtype)
    truncated = truncated.astype(obs.dtype)

    # Empirical batch weights for the implicit diagonal metric Ξ̂_B.
    valid_weights = 1.0 - done.reshape(-1).astype(obs.dtype)
    if hparams.mask_truncated:
        valid_weights = valid_weights * (1.0 - truncated.reshape(-1).astype(obs.dtype))
    valid_weights = valid_weights / jnp.maximum(valid_weights.sum(), 1.0)
    weight_column = valid_weights[:, None]

    # Critic features are fixed inputs for DICE/SF training. BatchNorm affine parameters are trainable global DICE/representation parameters.
    bn_scale = jax.lax.stop_gradient(dice_params["batch_norm_phi_scale"])
    bn_bias = jax.lax.stop_gradient(dice_params["batch_norm_phi_bias"])
    phi_raw = jax.lax.stop_gradient(critic_ref(obs, behavior_action)["embed"])

    next_action = sample_actor_action(actor_ref, next_obs, next_action_key, discrete_actions)
    next_phi_raw = jax.lax.stop_gradient(critic_ref(next_obs, next_action)["embed"])
    joint_phi = batch_norm_phi(
        jnp.concatenate([phi_raw, next_phi_raw], axis=0),
        bn_scale,
        bn_bias,
    )
    phi, next_phi = jnp.split(joint_phi, 2, axis=0)

    nu = dice_params["sr_dice_nu"]
    successor_matrix = dice_params["sr_dice_successor"]

    # Trainable successor features: ψ(s,a) = φ(s,a)S.
    psi = phi @ successor_matrix
    next_psi_target = jax.lax.stop_gradient(next_phi @ successor_matrix)
    bootstrap = (1.0 - done.reshape(-1)) * (1.0 - truncated.reshape(-1))
    successor_target = phi + hparams.gamma * bootstrap[:, None] * next_psi_target
    successor_residual = psi - successor_target
    successor_loss = jnp.sum(valid_weights * 0.5 * jnp.sum(jnp.square(successor_residual), axis=-1))

    # For independently sampled s₀ ∼ d₀, draw a₀ ∼ πₖ(·|s₀) and evaluate ψ(s₀,a₀).
    start_action = sample_actor_action(actor_ref, initial_obs, start_action_key, discrete_actions)
    start_phi_raw = jax.lax.stop_gradient(critic_ref(initial_obs, start_action)["embed"])
    start_phi = batch_norm_phi(start_phi_raw, bn_scale, bn_bias)
    successor_start_phi = start_phi @ successor_matrix
    successor_start_mean = successor_start_phi.mean(axis=0)

    # Original SR-DICE least-squares ratio: ρ(s,a) = φ(s,a)^Tν.
    rho = phi @ nu

    # Ratio update must not move the successor-feature matrix.
    successor_start_mean = jax.lax.stop_gradient(successor_start_mean)
    ratio_loss = (
        0.5 * jnp.sum(valid_weights * jnp.square(rho))
        - (1.0 - hparams.gamma) * jnp.dot(nu, successor_start_mean)
    )
    gram = phi.T @ (weight_column * phi)

    metrics = {
        "sr_dice/total_dice_loss": ratio_loss + successor_loss,
        "sr_dice/ratio_loss": ratio_loss,
        "sr_dice/successor_loss": successor_loss,
        "sr_dice/rho_mean": rho.mean(),
    }
    return rho.reshape(batch.done.shape), metrics

def invariance_aux_loss(
    curr_emb: jax.Array,         # [B, d], live critic/BN feature output
    next_emb_target: jax.Array,  # [B, d], frozen target feature output
    weights: jax.Array,          # [B], Ξ̂_B = diag(weights)
    feature_dynamics: jax.Array, # [d, d], persistent trainable F
    stop_feature_dynamics: bool = True,
):
    """BatchNorm/F invariance loss with stop-gradient critic features.

    This is the critic-side auxiliary step:
        ½‖sg(Z⁺) − ZF‖²_{Ξ̂_B,Frob}.

    The caller passes Z with stop-gradient critic features, so gradients do not
    flow into φ. Gradients flow through the trainable BatchNorm affine parameters and, when stop_feature_dynamics=False, through F.
    """
    Z_curr = curr_emb
    Z_next = jax.lax.stop_gradient(next_emb_target)
    weights = jax.lax.stop_gradient(weights)
    if stop_feature_dynamics:
        feature_dynamics = jax.lax.stop_gradient(feature_dynamics)

    residual = Z_next - Z_curr @ feature_dynamics
    per_sample_loss = 0.5 * jnp.sum(jnp.square(residual), axis=-1)
    loss = jnp.sum(weights * per_sample_loss)

    return loss, feature_dynamics, {
        "sr_dice/inv_loss": loss,
    }

def load_bc_weights_to_actor(bc_checkpoint_path: str, jax_actor: nnx.Module) -> nnx.Module:
    """
    Load JAX BC pretrained weights into JAX actor.
    
    Loads a pickle file saved by pretrain-jax.py (nnx.split state) and
    updates the actor's parameters in-place.
    
    Args:
        bc_checkpoint_path: Path to BC checkpoint file (.pkl)
        jax_actor: JAX actor to load weights into
    
    Returns:
        Modified jax_actor with BC weights loaded
    """
    import pickle
    
    # Verify checkpoint exists
    if not os.path.exists(bc_checkpoint_path):
        logging.warning(f"Checkpoint not found at {bc_checkpoint_path}, using random initialization")
        return jax_actor
    
    logging.info(f"Loading BC weights from {bc_checkpoint_path}")
    
    try:
        with open(bc_checkpoint_path, 'rb') as f:
            saved_state = pickle.load(f)
        
        nnx.update(jax_actor, saved_state)
        logging.info(f"Successfully loaded JAX BC weights from {bc_checkpoint_path}")
        return jax_actor
        
    except Exception as e:
        logging.error(f"Failed to load BC weights: {e}")
        logging.warning("Using random initialization instead")
        import traceback
        traceback.print_exc()
        return jax_actor

def _split_online_offline_mean(values: jax.Array, source_is_offline: jax.Array | None):
    vals = jnp.asarray(values).reshape(-1)
    if source_is_offline is None:
        return vals.mean(), jnp.array(0.0, dtype=vals.dtype)
    offline_mask = jnp.asarray(source_is_offline).reshape(-1).astype(vals.dtype)
    online_mask = 1.0 - offline_mask
    online_count = online_mask.sum()
    offline_count = offline_mask.sum()
    online_mean = jnp.where(
        online_count > 0,
        (vals * online_mask).sum() / online_count,
        jnp.array(0.0, dtype=vals.dtype),
    )
    offline_mean = jnp.where(
        offline_count > 0,
        (vals * offline_mask).sum() / offline_count,
        jnp.array(0.0, dtype=vals.dtype),
    )
    return online_mean, offline_mean

class REPPOPolicy(nnx.Module):
    def __init__(
        self,
        base: nnx.Module,
        normalizer: Normalizer | None,
        normalization_state,
        eval: bool,
        action_space: Space,
    ):
        self.base = base
        self.normalizer = normalizer
        self.normalization_state = nnx.data(normalization_state)
        self._eval_mode = eval
        self.action_space = action_space

    def __call__(self, key: jax.Array, x: jax.Array, **kwargs):
        if self.normalizer is not None:
            x = self.normalizer.normalize(self.normalization_state, x)

        if self._eval_mode:
            action = self.base.det_action(x)
            return action, {}

        pi = self.base(x, **kwargs)
        action = pi.sample(seed=key)

        if isinstance(self.action_space, Box):
            action = action.clip(-0.999, 0.999)

        behavior_log_prob = pi.log_prob(action)
        return action, {"behavior_log_prob": behavior_log_prob}

def make_policy_fn(
    cfg: DictConfig, observation_space: Space, action_space: Space
) -> Callable[[REPPOTrainState, bool], Policy]:
    cfg = cfg.algorithm

    def policy_fn(train_state: REPPOTrainState, eval: bool) -> Policy:
        normalizer = Normalizer() if cfg.normalize_env else None
        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)

        policy = REPPOPolicy(
            base=actor_model,
            normalizer=normalizer if cfg.normalize_env else None,
            normalization_state=train_state.normalization_state,
            eval=eval,
            action_space=action_space,
        )
        policy.eval()

        # def policy(key: Key, obs: jax.Array, **kwargs) -> tuple[jax.Array, dict]:
        #     if train_state.normalization_state is not None:
        #         obs = normalizer.normalize(train_state.normalization_state, obs)

        #     if eval:
        #         action: jax.Array = actor_model.det_action(obs)
        #     else:
        #         pi = actor_model(obs, scale=offset)
        #         action = pi.sample(seed=key)

        #     if isinstance(action_space, Box):
        #         action = action.clip(-0.999, 0.999)

        #     return action, {}

        return policy

    return policy_fn


def make_dice_optimizers(hparams) -> dict[str, optax.GradientTransformation]:
    """Adam optimizers for the SR-DICE / representation parameters.

    Defined once so make_init_fn (optimizer-state init) and make_learner_fn
    (parameter updates) build identical transforms. Each parameter gets its own
    Adam state so their moment estimates and step counts do not interfere.
    """
    sr_dice_lr = float(getattr(hparams, "sr_dice_lr", 1e-3))
    batch_norm_phi_lr = float(getattr(hparams, "batch_norm_phi_lr", sr_dice_lr))
    feature_dynamics_lr = float(getattr(hparams, "feature_dynamics_lr", sr_dice_lr))
    return {
        "sr_dice_nu": optax.adam(sr_dice_lr),
        "sr_dice_successor": optax.adam(sr_dice_lr),
        "feature_dynamics_F": optax.adam(feature_dynamics_lr),
        "batch_norm_phi_scale": optax.adam(batch_norm_phi_lr),
        "batch_norm_phi_bias": optax.adam(batch_norm_phi_lr),
    }

def make_init_fn(
    cfg: DictConfig,
    observation_space: Space,
    action_space: Space,
) -> InitFn:
    hparams = cfg.algorithm
    discrete_actions = isinstance(action_space, Discrete)

    def init(key: Key):
        key, model_key = jax.random.split(key)
        rngs = nnx.Rngs(model_key)

        optim_fn = hydra.utils.instantiate(hparams.optimizer)
        base_lr = hparams.optimizer.learning_rate
        actor_lr = getattr(hparams, "actor_lr_start", base_lr)

        if hparams.max_grad_norm is not None:
            tx = optax.chain(optax.clip_by_global_norm(hparams.max_grad_norm), optim_fn)
        else:
            tx = optim_fn

        actor_optim = optax.adam(learning_rate=actor_lr)
        actor_tx = (
            optax.chain(optax.clip_by_global_norm(hparams.max_grad_norm), actor_optim)
            if hparams.max_grad_norm is not None
            else actor_optim
        )
        logging.info(
            f"Actor optimizer: fixed lr={actor_lr:.1e}, critic_lr={base_lr:.1e}"
        )

        if hparams.normalize_env:
            normalizer = Normalizer()
            norm_state = normalizer.init(
                jax.tree.map(
                    lambda x: jnp.zeros_like(x, dtype=float),  # type: ignore
                    observation_space.sample(key),
                )
            )
        else:
            norm_state = None

        actor, critic = hydra.utils.call(cfg.algorithm.network)(
            cfg=cfg,
            action_space=action_space,
            observation_space=observation_space,
            rngs=rngs,
        )

        # Print JAX actor structure
        logging.info("JAX Actor structure successfully created")

        # Load BC pretrained weights into actor's feature_encoder only if bc_indicator is True
        if hparams.bc_indicator:
            bc_checkpoint_path = getattr(hparams, "bc_checkpoint_path", None)
            if bc_checkpoint_path and os.path.exists(bc_checkpoint_path):
                logging.info(f"Loading BC actor weights from {bc_checkpoint_path}")
                actor = load_bc_weights_to_actor(bc_checkpoint_path, actor)
                # Reset dual variables to REPPO defaults — BC-tuned values cause soft_reward distortion and unstable value targets.
                actor.log_temperature.value = jnp.ones(1) * math.log(hparams.ent_start)
                actor.log_lagrangian.value = jnp.ones(1) * math.log(hparams.kl_start)
                logging.info(
                    f"Reset log_temperature to log({hparams.ent_start})={math.log(hparams.ent_start):.4f}, "
                    f"log_lagrangian to log({hparams.kl_start})={math.log(hparams.kl_start):.4f}"
                )
            else:
                logging.info("No BC actor checkpoint specified or found, using random initialization")
        else:
            logging.info("bc_indicator=False, using random initialization for actor")

        dummy_obs = jax.tree.map(lambda x: jnp.zeros((1,) + x.shape, dtype=jnp.float32), observation_space.sample(key))
        if discrete_actions:
            dummy_action = jnp.zeros((1,), dtype=jnp.int32)
        else:
            dummy_action = jnp.zeros((1,) + action_space.shape, dtype=jnp.float32)
        sr_dice_feature_dim = critic(dummy_obs, dummy_action)["embed"].shape[-1]
        sr_dice_nu = jnp.zeros((sr_dice_feature_dim,), dtype=jnp.float32)
        sr_dice_successor = jnp.eye(sr_dice_feature_dim, dtype=jnp.float32)
        feature_dynamics = jnp.eye(sr_dice_feature_dim, dtype=jnp.float32)
        batch_norm_phi_scale = jnp.ones((sr_dice_feature_dim,), dtype=jnp.float32)
        batch_norm_phi_bias = jnp.zeros((sr_dice_feature_dim,), dtype=jnp.float32)

        dice_params = {
            "sr_dice_nu": sr_dice_nu,
            "sr_dice_successor": sr_dice_successor,
            "feature_dynamics_F": feature_dynamics,
            "batch_norm_phi_scale": batch_norm_phi_scale,
            "batch_norm_phi_bias": batch_norm_phi_bias,
        }
        dice_optimizers = make_dice_optimizers(hparams)
        dice_opt_state = {
            name: dice_optimizers[name].init(dice_params[name]) for name in dice_params
        }

        return REPPOTrainState.create(
            graphdef=nnx.graphdef(actor),
            params=dice_params,
            # The top-level train-state optimizer is unused: DICE/representation
            # params are updated by their own per-parameter Adam optimizers.
            tx=optax.set_to_zero(),
            dice_opt_state=dice_opt_state,
            actor=nnx.TrainState.create(
                graphdef=nnx.graphdef(actor), params=nnx.state(actor), tx=actor_tx
            ),
            critic=nnx.TrainState.create(
                graphdef=nnx.graphdef(critic), params=nnx.state(critic), tx=tx
            ),
            target_critic=nnx.TrainState.create(
                graphdef=nnx.graphdef(critic), params=nnx.state(critic), tx=optax.set_to_zero()
            ),
            actor_target=nnx.TrainState.create(
                graphdef=nnx.graphdef(actor),
                params=nnx.state(actor),
                tx=optax.set_to_zero(),
            ),
            iteration=0,
            time_steps=0,
            normalization_state=norm_state,
            last_env_state=None,
            last_obs=None,
        )

    return init


def make_learner_fn(
    cfg: DictConfig, observation_space: Space, action_space: Space
) -> LearnerFn:
    normalizer = Normalizer() if cfg.algorithm.normalize_env else None
    hparams = cfg.algorithm
    data_type = getattr(hparams, 'data_type', 'expert')
    discrete_actions = isinstance(action_space, Discrete)
    d = action_space.shape[-1] if not discrete_actions else action_space.n

    def critic_loss_fn(params: nnx.Param,
        batch_norm_phi_scale: jax.Array,
        batch_norm_phi_bias: jax.Array,
        feature_dynamics: jax.Array,
        train_state: REPPOTrainState,
        minibatch: Transition,
    ):
        critic_model = nnx.merge(train_state.critic.graphdef, params)
        critic_model.train()
        critic_output = critic_model(minibatch.obs, minibatch.action)
        curr_emb_raw = critic_output["embed"]
        target_values = minibatch.extras["target_values"]

        if hparams.hl_gauss:
            target_cat = jax.vmap(utils.hl_gauss, in_axes=(0, None, None, None))(
                target_values, hparams.num_bins, hparams.vmin, hparams.vmax
            )
            critic_pred = critic_output["logits"]
            critic_update_loss = optax.softmax_cross_entropy(critic_pred, target_cat)
        else:
            critic_pred = critic_output["value"]
            critic_update_loss = optax.squared_error(
                critic_pred.reshape(-1, 1),
                target_values.reshape(-1, 1),
            )

        # BatchNorm is trainable. The replay mask is used only in the auxiliary losses through Ξ̂_B.
        batch_weights = 1.0 - minibatch.done.reshape(-1).astype(curr_emb_raw.dtype)
        if hparams.mask_truncated:
            batch_weights = batch_weights * (1.0 - minibatch.truncated.reshape(-1).astype(curr_emb_raw.dtype))
        batch_weights = batch_weights / jnp.maximum(batch_weights.sum(), 1.0)
        joint_emb = batch_norm_phi(
            jnp.concatenate(
                [
                    jax.lax.stop_gradient(curr_emb_raw),
                    jax.lax.stop_gradient(minibatch.extras["next_emb"]),
                ],
                axis=0,
            ),
            batch_norm_phi_scale,
            batch_norm_phi_bias,
        )
        curr_emb, next_emb = jnp.split(joint_emb, 2, axis=0)
        inv_loss, feature_dynamics, inv_metrics = invariance_aux_loss(
            curr_emb,
            next_emb,
            batch_weights,
            feature_dynamics,
            stop_feature_dynamics=False,
        )
        gershgorin_eps = float(getattr(hparams, "gershgorin_eps", 1e-5))
        gram = jax.lax.stop_gradient(curr_emb).T @ (
            jax.lax.stop_gradient(batch_weights)[:, None]
            * jax.lax.stop_gradient(curr_emb)
        )
        gershgorin_loss_mult = float(getattr(hparams, "gershgorin_loss_mult", 1.0))
        gershgorin_loss_value, _, gershgorin_metrics = gershgorin_loss(
            gram,
            feature_dynamics,
            gamma=hparams.gamma,
            eps=gershgorin_eps,
        )
        pred_rew = critic_output["pred_rew"]
        value = critic_output["value"]
        aux_rew_loss = optax.squared_error(pred_rew.reshape(-1), minibatch.reward.reshape(-1))
        rew_aux_loss = jnp.sum(batch_weights * aux_rew_loss)
        # Per-loss multipliers so each auxiliary term can be tuned or switched
        # off independently (set the multiplier to 0.0 to disable a term).
        inv_loss_mult = float(getattr(hparams, "inv_loss_mult", 1.0))
        rew_aux_loss_mult = float(getattr(hparams, "rew_aux_loss_mult", 1.0))
        aux_loss = (
            inv_loss_mult * inv_loss
            + gershgorin_loss_mult * gershgorin_loss_value
            + rew_aux_loss_mult * rew_aux_loss
        )

        source_is_offline = minibatch.extras.get("source_is_offline", None)
        # compute l2 error for logging
        critic_loss_arr = optax.squared_error(
            value,
            target_values,
        )
        critic_loss, critic_loss_offline = _split_online_offline_mean(
            critic_loss_arr, source_is_offline
        )

        # Critic bias and error variance (using n-step/Retrace targets as G_t)
        mc_error = value.reshape(-1) - target_values.reshape(-1)
        mc_bias, mc_bias_offline = _split_online_offline_mean(mc_error, source_is_offline)  # E[Q - G_t] (signed)
        mc_second_moment, mc_second_moment_offline = _split_online_offline_mean(
            jnp.square(mc_error), source_is_offline
        )
        mc_error_var = mc_second_moment - jnp.square(mc_bias)
        mc_error_var_offline = mc_second_moment_offline - jnp.square(mc_bias_offline)
        
        # TD error: r + γ·E_a'[Q(s',a')] - Q(s,a)
        td_error = (
            minibatch.extras["soft_reward"].reshape(-1)
            + hparams.gamma * minibatch.extras["next_policy_value"].reshape(-1)
            - minibatch.extras["action_value"].reshape(-1)
        )
        td_error_mean, td_error_mean_offline = _split_online_offline_mean(td_error, source_is_offline)
        target_mean, target_mean_offline = _split_online_offline_mean(target_values, source_is_offline)
        target_second_moment, target_second_moment_offline = _split_online_offline_mean(
            jnp.square(target_values), source_is_offline
        )
        target_var = target_second_moment - jnp.square(target_mean)
        target_var_offline = target_second_moment_offline - jnp.square(target_mean_offline)
       
        q_mean, q_mean_offline = _split_online_offline_mean(value, source_is_offline)
        
        mask_truncated = hparams.mask_truncated
        mask = (1.0 - minibatch.truncated) if mask_truncated else 1.0
        # PER: scale loss by importance-sampling weight to correct for sampling bias
        is_w = minibatch.extras.get("is_weight", None)
        per_scale = is_w.reshape(-1) if (data_type == 'PER' and is_w is not None) else 1.0
        td_loss = jnp.mean(
            per_scale * mask * critic_update_loss
        )
        # `td_loss_mult` scales the value/TD loss; `aux_loss_mult` scales the
        # (already per-term weighted) auxiliary losses as a whole.
        td_loss_mult = float(getattr(hparams, "td_loss_mult", 1.0))
        loss = td_loss_mult * td_loss + hparams.aux_loss_mult * aux_loss
        unmasked_critic_total_loss = jnp.mean(critic_update_loss) + hparams.aux_loss_mult * aux_loss
        # Unified critic diagnostics (overall means). The per-source `_offline`
        # split is only added when the batch actually mixes sources (expert path).
        critic_diag = {
            "critic_diag/critic_loss": critic_loss,
            "critic_diag/mc_bias": mc_bias,
            "critic_diag/mc_error_var": mc_error_var,
            "critic_diag/td_error_mean": td_error_mean,
            "critic_diag/target_mean": target_mean,
            "critic_diag/target_var": target_var,
            "critic_diag/q": q_mean,
        }
        if source_is_offline is not None:
            critic_diag.update({
                "critic_diag/critic_loss_offline": critic_loss_offline,
                "critic_diag/mc_bias_offline": mc_bias_offline,
                "critic_diag/mc_error_var_offline": mc_error_var_offline,
                "critic_diag/td_error_mean_offline": td_error_mean_offline,
                "critic_diag/target_mean_offline": target_mean_offline,
                "critic_diag/target_var_offline": target_var_offline,
                "critic_diag/q_offline": q_mean_offline,
            })
        return loss, dict(
            critic_update_loss=critic_update_loss,
            masked_critic_total_loss=loss,
            unmasked_critic_total_loss=unmasked_critic_total_loss,
            aux_loss=aux_loss,
            rew_aux_loss=rew_aux_loss,
            gershgorin_loss=gershgorin_loss_value,
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            **inv_metrics,
            **gershgorin_metrics,
            **critic_diag,
        )

    def actor_loss(
        params: nnx.Param, train_state: REPPOTrainState, minibatch: Transition
    ):
        critic_target_model = nnx.merge(
            train_state.critic.graphdef,
            train_state.critic.params,
        )
        actor_model = nnx.merge(train_state.actor.graphdef, params)
        actor_target_model = nnx.merge(
            train_state.actor.graphdef, train_state.actor_target.params
        )

        # set up models for training with batch norm
        actor_model.train()
        critic_target_model.eval()
        actor_target_model.eval()
        pi = actor_model(minibatch.obs)
        old_pi = actor_target_model(minibatch.obs)

        # policy KL constraint
        kl = compute_policy_kl(minibatch=minibatch, pi=pi, old_pi=old_pi)
        temperature = jnp.squeeze(actor_model.temperature())
        alpha = jax.lax.stop_gradient(temperature)
        if discrete_actions:
            critic_pred = critic_target_model(minibatch.obs)
            value = critic_pred["value"]
            actor_loss = jnp.sum(pi.probs * ((alpha * pi.logits) - value), axis=-1)
            entropy = pi.entropy()
            action_size_target = -math.log(d) * hparams.ent_target_mult
        else:
            if hparams.gradient_estimator == "score_based_gae":
                if hparams.scale_samples_with_action_d:
                    num_samples = 16 * d
                else:
                    num_samples = 16
                pred_action, aux_log_prob = pi.sample_and_log_prob(
                    seed=minibatch.extras["action_key"],
                    sample_shape=(num_samples,),  # WARNING: magic number
                )
                adv = (
                    minibatch.extras["target_advs"]
                    - minibatch.extras["target_advs"].mean()
                ) / (minibatch.extras["target_advs"].std() + 1e-8)
                log_prob = pi.log_prob(minibatch.action.clip(-0.999, 0.999))
                old_log_prob = minibatch.extras["log_prob"]
                ratio = jnp.exp(log_prob - old_log_prob)
                actor_loss1 = ratio * adv
                EPS = 0.2  # hardcoded for now
                actor_loss2 = jnp.clip(ratio, 1.0 - EPS, 1.0 + EPS) * adv
                actor_loss = alpha * aux_log_prob.mean(0) - jnp.minimum(
                    actor_loss1, actor_loss2
                )
                entropy = -aux_log_prob.mean(axis=0)

            elif hparams.gradient_estimator == "score_based_q":
                if hparams.scale_samples_with_action_d:
                    num_samples = 4 * d
                else:
                    num_samples = 4
                pred_action, log_prob = pi.sample_and_log_prob(
                    seed=minibatch.extras["action_key"],
                    sample_shape=(num_samples,),  # WARNING: magic number
                )
                obs = jnp.repeat(minibatch.obs[None, ...], pred_action.shape[0], axis=0)
                critic_pred = critic_target_model(obs, pred_action)
                value = critic_pred["value"].sum(axis=0, keepdims=True)
                value = (value - critic_pred["value"]) / (
                    critic_pred["value"].shape[0] - 1
                )
                adv = critic_pred["value"] - value
                actor_loss = -jnp.mean(
                    log_prob * jax.lax.stop_gradient(adv) - alpha * log_prob, axis=0
                )
                entropy = -log_prob.mean(axis=0)

            elif hparams.gradient_estimator == "pathwise_q":
                pred_action, log_prob = pi.sample_and_log_prob(
                    seed=minibatch.extras["action_key"]
                )
                obs = minibatch.obs
                critic_pred = critic_target_model(obs, pred_action)
                value = critic_pred["value"]
                actor_loss = log_prob * alpha - value
                entropy = -log_prob

            else:
                raise ValueError(
                    f"Unknown gradient estimator: {hparams.gradient_estimator}"
                )
            action_size_target = d * hparams.ent_target_mult

        # Use the fitted SR-DICE ratio only for the policy-improvement term.
        # ρ̂ is detached: actor gradients do not update DICE and DICE gradients do not update actor.
        rho = jax.lax.stop_gradient(minibatch.extras["sr_dice_ratio"].reshape(-1))
        rho = jnp.clip(rho, a_min=1e-5, a_max=None)
        # Self-normalize ρ so the effective actor step size does not scale with
        # E_D[ρ] (the SR-DICE solution has an uncalibrated magnitude). After this
        # the weights average to 1, so the policy-improvement gradient magnitude
        # is decoupled from the absolute scale of the fitted ratio.
        if getattr(hparams, "normalize_sr_dice_ratio", True):
            rho = rho / (jnp.mean(rho) + 1e-8)
        dice_actor_loss = jnp.mean(rho * actor_loss.reshape(-1))

        # Keep KL/entropy unchanged
        kl_mean = kl.reshape(-1).mean()
        entropy_mean = entropy.reshape(-1).mean()
        target_entropy = action_size_target + entropy
        target_entropy_mean = target_entropy.reshape(-1).mean()

        lagrangian = actor_model.lagrangian()
        if hparams.actor_kl_clip_mode == "full":
            loss = dice_actor_loss + kl_mean * jax.lax.stop_gradient(lagrangian) * hparams.reduce_kl
        elif hparams.actor_kl_clip_mode == "clipped":
            kl_penalty = jnp.where(
                kl.reshape(-1) < hparams.kl_bound,
                0.0,
                kl.reshape(-1) * jax.lax.stop_gradient(lagrangian) * hparams.reduce_kl,
            ).mean()
            loss = dice_actor_loss + kl_penalty
        elif hparams.actor_kl_clip_mode == "value":
            loss = dice_actor_loss
        else:
            raise ValueError(f"Unknown actor loss mode: {hparams.actor_kl_clip_mode}")

        target_entropy_loss = temperature * jax.lax.stop_gradient(target_entropy_mean)
        lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl_mean - hparams.kl_bound)

        if hparams.update_entropy_lagrangian:
            loss = loss + target_entropy_loss
        if hparams.update_kl_lagrangian:
            loss = loss + lagrangian_loss

        loss = jnp.squeeze(loss)

        real_action_log_prob = old_pi.log_prob(clip_action_for_critic(minibatch.action, action_space)).mean()
        
        return loss, dict(
            temp=temperature,
            abs_pred_action=jnp.abs(pred_action).mean() if not discrete_actions else 0.0,
            lagrangian=lagrangian,
            lagrangian_loss=lagrangian_loss,
            entropy_loss=target_entropy_loss,
            **{
                "actor_diag/kl": kl_mean,
                "actor_diag/entropy": entropy_mean,
                "actor_diag/actor_loss": dice_actor_loss,
                "actor_diag/sr_dice_ratio_mean": rho.mean(),
                "actor_diag/real_action_log_prob": real_action_log_prob,
            },
        )

    def compute_policy_kl(
        minibatch: Transition, pi: distrax.Distribution, old_pi: distrax.Distribution
    ) -> jax.Array:
        if hparams.reverse_kl:
            if discrete_actions:
                kl = pi.kl_divergence(old_pi)
            else:
                pi_action, pi_act_log_prob = pi.sample_and_log_prob(
                    sample_shape=(4,), seed=minibatch.extras["kl_key"]
                )
                pi_action = jnp.clip(pi_action, -1 + 1e-4, 1 - 1e-4)
                old_pi_act_log_prob = old_pi.log_prob(pi_action).mean(0)
                pi_act_log_prob = pi_act_log_prob.mean(0)
                kl = pi_act_log_prob - old_pi_act_log_prob
        else:
            if discrete_actions:
                kl = old_pi.kl_divergence(pi)
            else:
                old_pi_action, old_pi_act_log_prob = old_pi.sample_and_log_prob(
                    sample_shape=(4,), seed=minibatch.extras["kl_key"]
                )
                old_pi_action = jnp.clip(old_pi_action, -1 + 1e-4, 1 - 1e-4)

                old_pi_act_log_prob = old_pi_act_log_prob.mean(0)
                pi_act_log_prob = pi.log_prob(old_pi_action).mean(0)
                kl = old_pi_act_log_prob - pi_act_log_prob
        return kl

    def polyak_update_target_critic(train_state: REPPOTrainState):
        polyak = hparams.polyak
        target_params = jax.tree.map(
            lambda target, live: (1.0 - polyak) * target + polyak * live,
            train_state.target_critic.params,
            train_state.critic.params,
        )
        return train_state.replace(target_critic=train_state.target_critic.replace(params=target_params))

    dice_optimizers = make_dice_optimizers(hparams)

    def apply_dice_update(state, name, grad):
        """Adam update for a single SR-DICE / representation parameter."""
        optimizer = dice_optimizers[name]
        opt_state = state.dice_opt_state[name]
        updates, new_opt_state = optimizer.update(grad, opt_state, state.params[name])
        new_param = optax.apply_updates(state.params[name], updates)
        new_params = {**state.params, name: new_param}
        new_dice_opt_state = {**state.dice_opt_state, name: new_opt_state}
        return state.replace(params=new_params, dice_opt_state=new_dice_opt_state)

    def critic_update(train_state: REPPOTrainState, minibatch: Transition):
        critic_grad_fn = jax.value_and_grad(critic_loss_fn, argnums=(0, 1, 2, 3), has_aux=True)
        output, (critic_grads, bn_scale_grad, bn_bias_grad, feature_dynamics_grad) = critic_grad_fn(
            train_state.critic.params,
            train_state.params["batch_norm_phi_scale"],
            train_state.params["batch_norm_phi_bias"],
            train_state.params["feature_dynamics_F"],
            train_state,
            minibatch,
        )
        critic_grads = jax.tree.map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), critic_grads)
        bn_scale_grad = jnp.nan_to_num(bn_scale_grad, nan=0.0, posinf=1.0, neginf=-1.0)
        bn_bias_grad = jnp.nan_to_num(bn_bias_grad, nan=0.0, posinf=1.0, neginf=-1.0)
        feature_dynamics_grad = jnp.nan_to_num(feature_dynamics_grad, nan=0.0, posinf=1.0, neginf=-1.0)

        critic_train_state = train_state.critic.apply_gradients(critic_grads)
        train_state = train_state.replace(critic=critic_train_state)

        train_state = apply_dice_update(train_state, "batch_norm_phi_scale", bn_scale_grad)
        train_state = apply_dice_update(train_state, "batch_norm_phi_bias", bn_bias_grad)
        train_state = apply_dice_update(train_state, "feature_dynamics_F", feature_dynamics_grad)

        return polyak_update_target_critic(train_state), output[1]

    def actor_update(train_state: REPPOTrainState, minibatch: Transition):
        def update_actor(_):
            actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
            output, grads = actor_grad_fn(train_state.actor.params, train_state, minibatch)
            grad_norm = jax.tree.reduce(operator.add, jax.tree.map(lambda x: jnp.linalg.norm(x), grads))
            grads = jax.tree.map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), grads)
            return train_state.actor.apply_gradients(grads), grad_norm, output[1]

        def hold_actor(_):
            _, actor_metrics = actor_loss(train_state.actor.params, train_state, minibatch)
            return train_state.actor, jnp.array(0.0), actor_metrics

        if cfg.algorithm.bc_indicator:
            delay = getattr(hparams, "bc_actor_update_delay", 0)
            actor_train_state, grad_norm, actor_metrics = jax.lax.cond(train_state.iteration > delay, update_actor, hold_actor, None)
        else:
            actor_train_state, grad_norm, actor_metrics = update_actor(None)
        return train_state.replace(actor=actor_train_state), {**actor_metrics, "actor_diag/grad_norm": grad_norm}

    # Train/log SR-DICE from replay minibatches, like a separate critic-style estimator.
    # Both DICE components are persistent linear layers:
    #   sr_dice_successor: ψ(s,a) = φ_bn(s,a)S
    #   sr_dice_nu:       ρ(s,a) = φ_bn(s,a)^Tν
    def dice_update(train_state: REPPOTrainState, minibatch: Transition, initial_obs: jax.Array):
        dice_key = minibatch.extras["dice_key"]

        # 1) Successor-feature update: update S only.
        def successor_loss_fn(successor_matrix):
            dice_params = {
                **train_state.params,
                "sr_dice_nu": jax.lax.stop_gradient(train_state.params["sr_dice_nu"]),
                "sr_dice_successor": successor_matrix,
                "feature_dynamics_F": jax.lax.stop_gradient(train_state.params["feature_dynamics_F"]),
                "batch_norm_phi_scale": jax.lax.stop_gradient(train_state.params["batch_norm_phi_scale"]),
                "batch_norm_phi_bias": jax.lax.stop_gradient(train_state.params["batch_norm_phi_bias"]),
            }
            _, dice_metrics = fit_sr_dice_ratio(
                dice_key, dice_params, train_state, minibatch,
                initial_obs, hparams, action_space, discrete_actions
            )
            return dice_metrics["sr_dice/successor_loss"], dice_metrics

        (successor_loss, successor_metrics), successor_grad = jax.value_and_grad(successor_loss_fn, has_aux=True)(
            train_state.params["sr_dice_successor"]
        )
        successor_grad = jnp.nan_to_num(successor_grad, nan=0.0, posinf=1.0, neginf=-1.0)
        successor_grad_norm = jnp.linalg.norm(successor_grad)
        train_state = apply_dice_update(train_state, "sr_dice_successor", successor_grad)

        # 2) Ratio update: update ν only. S is fixed for this step.
        def ratio_loss_fn(nu):
            dice_params = {
                **train_state.params,
                "sr_dice_nu": nu,
                "sr_dice_successor": jax.lax.stop_gradient(train_state.params["sr_dice_successor"]),
                "feature_dynamics_F": jax.lax.stop_gradient(train_state.params["feature_dynamics_F"]),
                "batch_norm_phi_scale": jax.lax.stop_gradient(train_state.params["batch_norm_phi_scale"]),
                "batch_norm_phi_bias": jax.lax.stop_gradient(train_state.params["batch_norm_phi_bias"]),
            }
            _, dice_metrics = fit_sr_dice_ratio(
                dice_key, dice_params, train_state, minibatch,
                initial_obs, hparams, action_space, discrete_actions
            )
            return dice_metrics["sr_dice/ratio_loss"], dice_metrics

        (ratio_loss, ratio_metrics), ratio_grad = jax.value_and_grad(ratio_loss_fn, has_aux=True)(
            train_state.params["sr_dice_nu"]
        )
        ratio_grad = jnp.nan_to_num(ratio_grad, nan=0.0, posinf=1.0, neginf=-1.0)
        ratio_grad_norm = jnp.linalg.norm(ratio_grad)
        train_state = apply_dice_update(train_state, "sr_dice_nu", ratio_grad)

        dice_metrics = {
            **ratio_metrics,
            "sr_dice/successor_loss": successor_loss,
            "sr_dice/ratio_loss": ratio_loss,
            "sr_dice/total_dice_loss": successor_loss + ratio_loss,
            "sr_dice/grad_norm": successor_grad_norm + ratio_grad_norm,
        }
        return train_state, dice_metrics

    def run_epoch(
        key: jax.Array, train_state: REPPOTrainState, batch: Transition, update_fn
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        key, shuffle_key, act_key, kl_key, dice_key = jax.random.split(key, 5)

        batch_size = batch.obs.shape[0] * batch.obs.shape[1]
        mini_batch_size = batch_size // hparams.num_mini_batches

        # [T, B, ...] -> [T * B, ...].
        flat_batch = jax.tree.map(
            lambda x: x.reshape((batch_size, *x.shape[2:])),
            batch,
        )

        indices = jax.random.permutation(shuffle_key, batch_size)
        minibatch_idxs = indices.reshape(
            hparams.num_mini_batches,
            mini_batch_size,
        )

        minibatches = jax.tree.map(
            lambda x: jnp.take(x, minibatch_idxs, axis=0),
            flat_batch,
        )
        minibatches = minibatches.replace(
            extras={
                **minibatches.extras,
                "action_key": jax.random.split(act_key, hparams.num_mini_batches),
                "kl_key": jax.random.split(kl_key, hparams.num_mini_batches),
                "dice_key": jax.random.split(dice_key, hparams.num_mini_batches),
            }
        )

        train_state, metrics = jax.lax.scan(update_fn, train_state, minibatches)

        metrics_mean = jax.tree.map(lambda x: x.mean(0), metrics)
        if "actor_diag/grad_norm" in metrics:
            metrics_mean["actor_diag/grad_norm_var"] = jnp.var(metrics["actor_diag/grad_norm"])
        return train_state, metrics_mean

    def nstep_lambda(batch: Transition):
        if not getattr(hparams, "use_retrace", True):
            def loop(carry: tuple[jax.Array, ...], transition: Transition):
                lambda_return, gae, truncated, next_value = carry

                truncated = transition.truncated
                done = transition.done
                reward = transition.extras["soft_reward"]
                value = transition.extras["value"]
                policy_value = transition.extras["policy_value"]
                lambda_sum = hparams.lmbda * lambda_return + (1 - hparams.lmbda) * value
                lambda_return = reward + hparams.gamma * jnp.where(
                    truncated, value, (1.0 - done) * lambda_sum
                )

                # GAE for policy
                delta = reward + hparams.gamma * (1.0 - done) * next_value - policy_value
                gae = delta + hparams.gamma * (1.0 - done) * hparams.lmbda * gae
                truncated_gae = reward + hparams.gamma * (1.0 - done) * next_value - value
                gae = jnp.where(truncated, truncated_gae, gae)

                return (
                    lambda_return,
                    gae,
                    truncated,
                    policy_value,
                ), (lambda_return, gae)

            _, (target_values, target_advs) = jax.lax.scan(
                f=loop,
                init=(
                    batch.extras["value"][-1],
                    batch.extras["policy_value"][-1],
                    jnp.ones_like(batch.truncated[0]),
                    batch.extras["policy_value"][-1],
                ),
                xs=batch,
                reverse=True,
            )

            retrace_coeff_mean = jnp.array(0.0, dtype=target_values.dtype)

        else:
            # Retrace(lambda) for replayed behavior-policy data.
            def loop(carry: tuple[jax.Array, ...], transition: Transition):
                lambda_return, q_next, retrace_coeff_next, next_value, gae = carry

                done = transition.done
                reward = transition.extras["soft_reward"]
                behavior_log_prob = transition.extras["behavior_log_prob"]
                current_log_prob = transition.extras["current_log_prob"]
                policy_value = transition.extras["policy_value"]
                action_value = transition.extras["action_value"]
                truncated = transition.truncated
                c_t_raw = hparams.lmbda * jnp.minimum(
                    1.0, jnp.exp(current_log_prob - behavior_log_prob)
                )
                c_t = jnp.where(truncated, 0.0, c_t_raw)
                # G_t = r_tilde[t] + gamma * (1 - d[t]) *
                #       (V[t+1] + c[t+1] * (G[t+1] - Q(x[t+1], a[t+1])))
                lambda_return = reward + hparams.gamma * jnp.where(
                    truncated,
                    next_value,
                    (1.0 - done.astype(jnp.float32))
                    * (next_value + retrace_coeff_next * (lambda_return - q_next)),
                )

                # GAE calculation
                delta = reward + hparams.gamma * (1.0 - done.astype(jnp.float32)) * next_value - policy_value
                gae = delta + hparams.gamma * (1.0 - done.astype(jnp.float32)) * hparams.lmbda * gae
                truncated_gae = delta
                gae = jnp.where(truncated, truncated_gae, gae)

                return (
                    lambda_return,
                    action_value,
                    c_t,
                    policy_value,
                    gae,
                ), (lambda_return, gae, c_t)

            _, (target_values, target_advs, retrace_coeffs) = jax.lax.scan(
                f=loop,
                init=(
                    batch.extras["next_policy_value"][-1],
                    batch.extras["action_value"][-1],
                    jnp.zeros_like(batch.extras["value"][-1]),
                    batch.extras["target_next_policy_value"][-1],
                    batch.extras["policy_value"][-1],
                ),
                xs=batch,
                reverse=True,
            )

            retrace_coeff_mean = retrace_coeffs.mean()

        return target_values, target_advs, retrace_coeff_mean

    def compute_extras(key: Key, train_state: REPPOTrainState, batch: Transition):
        key, act1_key, act2_key, act3_key = jax.random.split(key, 4)

        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        target_critic_model = nnx.merge(
            train_state.target_critic.graphdef,
            train_state.target_critic.params,
        )
        actor_model.eval()
        critic_model.eval()
        target_critic_model.eval()

        # TD-lambda bootstrap values come from the target critic.
        next_action, next_log_prob = actor_model(batch.next_obs).sample_and_log_prob(
            seed=act1_key
        )
        target_next_output = target_critic_model(batch.next_obs, next_action)
        live_next_output = critic_model(batch.next_obs, next_action)
        target_value = target_next_output["value"]
        next_emb = live_next_output["embed"]

        if hparams.scale_samples_with_action_d:
            num_samples = 8 * d
        else:
            num_samples = 8

        critic_action = batch.action.clip(-0.999, 0.999) if isinstance(action_space, Box) else batch.action
        action_value = critic_model(batch.obs, critic_action)["value"]

        next_pi = actor_model(batch.next_obs)
        next_actions = next_pi.sample(seed=act3_key, sample_shape=(num_samples,))
        next_actions = jnp.clip(next_actions, -0.999, 0.999)
        next_obs_tiled = jnp.repeat(
            batch.next_obs[None, ...], next_actions.shape[0], axis=0
        )
        next_policy_value = critic_model(next_obs_tiled, next_actions)["value"].mean(0)
        target_next_policy_value = target_critic_model(next_obs_tiled, next_actions)["value"].mean(0)

        soft_reward = batch.reward - hparams.gamma * next_log_prob * actor_model.temperature()

        pi = actor_model(batch.obs)
        current_log_probs = pi.log_prob(batch.action.clip(-0.999, 0.999))
        policy_actions = pi.sample(seed=act2_key, sample_shape=(num_samples,))
        policy_actions = jnp.clip(policy_actions, -0.999, 0.999)
        obs_tiled = jnp.repeat(batch.obs[None, ...], policy_actions.shape[0], axis=0)
        policy_value = critic_model(obs_tiled, policy_actions)["value"].mean(0)
        
        return {
            "soft_reward": soft_reward * cfg.env.get("reward_scaling", 1.0),
            "value": target_value,
            "action_value": action_value,
            "policy_value": policy_value,
            "next_policy_value": next_policy_value,
            "target_next_policy_value": target_next_policy_value,
            "next_emb": next_emb,
            "log_prob": current_log_probs,
            "current_log_prob": current_log_probs,
        }

    def learner_fn(
        key: Key, train_state: REPPOTrainState, batch: Transition
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        if "initial_obs" not in batch.extras:
            raise KeyError("SR-DICE requires a separately sampled batch.extras['initial_obs'] with s₀ ∼ d₀.")
        initial_obs = batch.extras["initial_obs"]
        batch = batch.replace(extras={k: v for k, v in batch.extras.items() if k != "initial_obs"})
        if hparams.normalize_env:
            new_norm_state = normalizer.update(train_state.normalization_state, batch.obs)
            initial_obs = normalizer.normalize(train_state.normalization_state, initial_obs)
            batch = batch.replace(obs=normalizer.normalize(train_state.normalization_state, batch.obs), next_obs=normalizer.normalize(train_state.normalization_state, batch.next_obs))
            train_state = train_state.replace(normalization_state=new_norm_state)

        # Freeze πₖ once: critic targets, SR-DICE, and every actor minibatch use this reference policy.
        train_state = train_state.replace(actor_target=train_state.actor_target.replace(params=train_state.actor.params))
        key, diag_key = jax.random.split(key)

        diagnostics_extras = compute_extras(key=diag_key, train_state=train_state, batch=batch)
        batch = batch.replace(extras={**batch.extras, **diagnostics_extras})

        target_values, target_advs, retrace_coeff_mean = nstep_lambda(batch)
        batch = batch.replace(extras={**batch.extras, "target_values": target_values, "target_advs": target_advs})

        current_log_prob = batch.extras["current_log_prob"]
        reward_mean = batch.reward.mean()
        policy_log_prob_mean = current_log_prob.mean()
        source_is_offline = batch.extras.get("source_is_offline")
        if source_is_offline is not None:
            offline_count = source_is_offline.sum()
            online_mask = 1.0 - source_is_offline
            online_count = online_mask.sum()
            reward_mean_offline = jnp.where(offline_count > 0, (batch.reward * source_is_offline).sum() / offline_count, 0.0)
            reward_mean_online = jnp.where(online_count > 0, (batch.reward * online_mask).sum() / online_count, 0.0)
            policy_log_prob_offline = jnp.where(offline_count > 0, (current_log_prob * source_is_offline).sum() / offline_count, 0.0)
            policy_log_prob_online = jnp.where(online_count > 0, (current_log_prob * online_mask).sum() / online_count, 0.0)
        else:
            reward_mean_offline = jnp.array(0.0, dtype=reward_mean.dtype)
            reward_mean_online = reward_mean
            policy_log_prob_offline = jnp.array(0.0, dtype=policy_log_prob_mean.dtype)
            policy_log_prob_online = policy_log_prob_mean

        per_env_td_error = jnp.abs(batch.extras["action_value"] - batch.extras["target_values"]).mean(axis=0)
        key, critic_key, sr_dice_key, ratio_key, pv_before_key, actor_key, pv_after_key = jax.random.split(key, 7)

        # Critic/representation phase: πₖ is fixed and θ is not updated.
        train_state, critic_metrics = jax.lax.scan(
            lambda state, epoch_key: run_epoch(epoch_key, state, batch, critic_update),
            train_state,
            jax.random.split(critic_key, 1),
        )
        critic_metrics = jax.tree.map(lambda x: x[-1], critic_metrics)

        # Train SR-DICE on replay minibatches with π_k fixed through actor_target.
        # This is amortized training: ν is updated from the buffer and carried across learner calls.
        train_state, dice_metrics = jax.lax.scan(
            lambda state, epoch_key: run_epoch(
                epoch_key,
                state,
                batch,
                lambda s, mb: dice_update(s, mb, initial_obs),
            ),
            train_state,
            jax.random.split(sr_dice_key, 1),
        )
        dice_metrics = jax.tree.map(lambda x: x[-1], dice_metrics)

        # Compute the latest fitted ratio once, detach it, and pass it to the actor phase.
        sr_dice_ratio, _ = fit_sr_dice_ratio(ratio_key, train_state.params, train_state, batch, initial_obs, hparams, action_space, discrete_actions)
        batch = batch.replace(
            extras={
                **batch.extras,
                "sr_dice_ratio": jax.lax.stop_gradient(sr_dice_ratio),
            }
        )

        # Policy Improvement Logging - Evaluate πₖ and πₖ₊₁ with the same frozen post-critic Q
        _n_pv = 8 * d if hparams.scale_samples_with_action_d else 8
        actor_before = nnx.merge(train_state.actor.graphdef, train_state.actor_target.params)
        critic_before = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        
        actor_before.eval()
        critic_before.eval()
        
        action_before = actor_before(batch.obs).sample(seed=pv_before_key, sample_shape=(_n_pv,))
        action_before = clip_action_for_critic(action_before, action_space)
        obs_tiled = jnp.repeat(batch.obs[None, ...], action_before.shape[0], axis=0)
        policy_value_before_vec = critic_before(obs_tiled, action_before)["value"].mean(0).reshape(-1)

        # actor-only policy-improvement phase with sg[ρ̂ₖ(s, aᵇ)].
        train_state, actor_metrics = jax.lax.scan(
            lambda state, epoch_key: run_epoch(epoch_key, state, batch, actor_update),
            train_state,
            jax.random.split(actor_key, 1),
        )
        actor_metrics = jax.tree.map(lambda x: x[-1], actor_metrics)

        actor_after = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_after = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        actor_after.eval()
        critic_after.eval()
        action_after = actor_after(batch.obs).sample(seed=pv_after_key, sample_shape=(_n_pv,))
        action_after = clip_action_for_critic(action_after, action_space)
        obs_tiled = jnp.repeat(batch.obs[None, ...], action_after.shape[0], axis=0)
        policy_value_after_vec = critic_after(obs_tiled, action_after)["value"].mean(0).reshape(-1)
        policy_improvement_online, policy_improvement_offline = _split_online_offline_mean(policy_value_after_vec - policy_value_before_vec, source_is_offline)

        base_metrics = {
            "reward_mean": reward_mean,
            "actor_diag/policy_log_prob_mean": policy_log_prob_mean,
            "critic_diag/retrace_coeff_mean": retrace_coeff_mean,
            "actor_diag/retrace_coeff_mean": retrace_coeff_mean,
            "actor_diag/policy_improvement": policy_improvement_online,
            "sys/grad_updates": (train_state.time_steps // (hparams.num_steps * hparams.num_envs)) * 2 * 1 * hparams.num_mini_batches,
        }
        if source_is_offline is not None:
            base_metrics.update({
                "reward_mean_offline": reward_mean_offline,
                "reward_mean_online": reward_mean_online,
                "actor_diag/policy_log_prob_offline_replay": policy_log_prob_offline,
                "actor_diag/policy_log_prob_online_replay": policy_log_prob_online,
                "actor_diag/policy_improvement_offline": policy_improvement_offline,
            })
        return train_state, {**critic_metrics, **actor_metrics, **dice_metrics, **base_metrics}, per_env_td_error

    return jax.jit(learner_fn)