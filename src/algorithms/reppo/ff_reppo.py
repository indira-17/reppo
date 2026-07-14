import logging
import math
from typing import Callable
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

def clip_action_for_critic(action: jax.Array, action_space: Space) -> jax.Array:
    return action.clip(-0.999, 0.999) if isinstance(action_space, Box) else action

def sample_actor_action(actor_model: nnx.Module, obs: jax.Array, key: jax.Array, discrete_actions: bool) -> jax.Array:
    pi = actor_model(obs)
    if discrete_actions:
        return pi.sample(seed=key)
    action, _ = pi.sample_and_log_prob(seed=key)
    return action.clip(-0.999, 0.999)

def gershgorin_loss(
    curr_features: jax.Array,
    next_features: jax.Array,
    weights: jax.Array,
    continuation: jax.Array,
    gamma: float,
    eps: float = 1e-5,
):
    """Gershgorin loss on the empirical TD iteration matrix.

    With row-wise features Φ_B ∈ R^{B×d}, the sampled matrix is

        A_B = Φ_Bᵀ Ξ_B (Φ_B - γ P^πΦ_B)
            ≈ φ(s,a)ᵀ Ξ_B [φ(s,a) - γφ(s',a')].

    Gradients flow through both current and policy-next features, so this loss
    shapes Φ directly. No learned feature-dynamics matrix is used here.
    """
    weights = jax.lax.stop_gradient(weights)
    continuation = jax.lax.stop_gradient(continuation).reshape(-1, 1)
    td_features = curr_features - gamma * continuation * next_features
    td_matrix = curr_features.T @ (weights[:, None] * td_features)

    diag = jnp.diag(td_matrix)
    off_diag_radius = jnp.sum(jnp.abs(td_matrix), axis=-1) - jnp.abs(diag)
    margin = diag - off_diag_radius
    violation = jax.nn.relu(eps - margin)
    loss = jnp.sum(violation) / td_matrix.size

    return loss, td_matrix, {
        "sr_dice/gershgorin_loss": loss,
        "sr_dice/gershgorin_margin_min": margin.min(),
        "sr_dice/gershgorin_margin_mean": margin.mean(),
    }


def compute_successor_start_mean(dice_params: dict[str, jax.Array], start_phi: jax.Array):
    """Compute E_{d₀,π}[ψ(s₀,a₀)] from cached live-policy features."""
    successor = jax.lax.stop_gradient(dice_params["sr_dice_successor"])
    successor_start = jax.lax.stop_gradient(start_phi) @ successor
    return jax.lax.stop_gradient(successor_start.reshape((-1, successor_start.shape[-1])).mean(axis=0))


def fit_sr_dice_ratio(
    dice_params: dict[str, jax.Array],
    train_state: REPPOTrainState,
    batch: Transition,
    hparams,
    action_space: Space,
    successor_start_mean: jax.Array,
):
    """Evaluate SR-DICE from the cached shared feature φ_k(s,a)."""
    del train_state, action_space
    phi = jax.lax.stop_gradient(batch.extras["sr_dice_phi"])
    flat_phi = phi.reshape((-1, phi.shape[-1]))
    done = batch.done.reshape(-1)
    truncated = batch.truncated.reshape(-1)
    valid_weights = jnp.ones_like(done, dtype=flat_phi.dtype)
    if hparams.mask_truncated:
        valid_weights = valid_weights * (1.0 - truncated.astype(flat_phi.dtype))
    valid_weights = valid_weights / jnp.maximum(valid_weights.sum(), 1.0)
    nu = dice_params["sr_dice_nu"]
    rho = flat_phi @ nu
    successor_start_mean = jax.lax.stop_gradient(successor_start_mean)
    ratio_loss = 0.5 * jnp.sum(valid_weights * jnp.square(rho)) - (1.0 - hparams.gamma) * jnp.dot(nu, successor_start_mean)
    rho_flat = rho.reshape(-1)
    # ESS is defined for non-negative importance weights. Use the same positive
    # clipping as the actor ratio, and exclude transitions masked from the
    # SR-DICE objective. This is the raw ESS in [1, number of valid samples].
    valid_mask = valid_weights > 0.0
    rho_ess_weights = jnp.where(
        valid_mask,
        jnp.clip(rho_flat, a_min=1e-5),
        jnp.zeros_like(rho_flat),
    )
    rho_ess = (jnp.sum(rho_ess_weights) ** 2) / (
        jnp.sum(jnp.square(rho_ess_weights)) + 1e-8
    )
    metrics = {
        "sr_dice/ratio_loss": ratio_loss,
        "sr_dice/rho_mean": rho_flat.mean(),
        "sr_dice/rho_std": rho_flat.std(),
        "sr_dice/rho_ess": rho_ess,
    }
    return rho.reshape(batch.done.shape), metrics

def invariance_aux_loss(
    pred_features: jax.Array,
    next_features_target: jax.Array,
    weights: jax.Array,
):
    """REPPO feature-prediction objective in the shared normalized φ-space.

        L_inv = 1/2 E_{ξ_k}[||f_φ(s,a) - sg(φ(s',a'))||²].

    The prediction and target inputs use the same saved normalization state and
    affine parameters used by Gershgorin, successor features, and SR-DICE.
    """
    target = jax.lax.stop_gradient(next_features_target)
    weights = jax.lax.stop_gradient(weights)
    residual = pred_features - target
    per_sample_loss = 0.5 * jnp.sum(jnp.square(residual), axis=-1)
    loss = jnp.sum(weights * per_sample_loss)
    return loss, {"sr_dice/inv_loss": loss}


def successor_feature_td_loss(
    curr_features: jax.Array,
    next_features: jax.Array,
    weights: jax.Array,
    continuation: jax.Array,
    successor: jax.Array,
    gamma: float,
):
    """Semi-gradient TD loss for ψ_k(s,a)ᵀ = φ_k(s,a)ᵀS_k."""
    curr_features = jax.lax.stop_gradient(curr_features)
    next_features = jax.lax.stop_gradient(next_features)
    weights = jax.lax.stop_gradient(weights)
    continuation = jax.lax.stop_gradient(continuation).reshape(-1, 1)

    successor_features = curr_features @ successor
    next_successor_features = next_features @ successor
    target = jax.lax.stop_gradient(
        curr_features + gamma * continuation * next_successor_features
    )
    residual = successor_features - target
    per_sample_loss = 0.5 * jnp.sum(jnp.square(residual), axis=-1)
    loss = jnp.sum(weights * per_sample_loss)

    return loss, {"sr_dice/successor_loss": loss}

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
    sr_dice_lr = float(getattr(hparams, "sr_dice_lr", 1e-3))
    optimizers = {"sr_dice_nu": optax.adam(sr_dice_lr), "sr_dice_successor": optax.adam(sr_dice_lr)}
    if bool(getattr(hparams, "train_batch_norm_phi", False)):
        optimizers["batch_norm_phi"] = optax.adam(float(getattr(hparams, "batch_norm_phi_lr", sr_dice_lr)))
    return optimizers

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
       
        batch_norm_phi = nnx.BatchNorm(sr_dice_feature_dim, rngs=rngs)
        batch_norm_phi_graphdef, batch_norm_phi_params, batch_norm_phi_stats = nnx.split(batch_norm_phi, nnx.Param, nnx.BatchStat)
        
        dice_params = {"sr_dice_nu": sr_dice_nu, "sr_dice_successor": sr_dice_successor, "batch_norm_phi": batch_norm_phi_params, "batch_norm_phi_stats": batch_norm_phi_stats}
        dice_optimizers = make_dice_optimizers(hparams)
        dice_opt_state = {name: optimizer.init(dice_params[name]) for name, optimizer in dice_optimizers.items()}

        return REPPOTrainState.create(
            graphdef=batch_norm_phi_graphdef,
            params=dice_params,
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

    def critic_loss_fn(params: nnx.Param, batch_norm_phi_params, batch_norm_phi_stats, train_state: REPPOTrainState, minibatch: Transition):
        critic_model = nnx.merge(train_state.critic.graphdef, params)
        critic_model.train()

        next_action = clip_action_for_critic(minibatch.extras["next_action"], action_space)
        joint_obs = jnp.concatenate([minibatch.obs, minibatch.next_obs], axis=0)
        joint_action = jnp.concatenate([minibatch.action, next_action], axis=0)
        joint_output = critic_model(joint_obs, joint_action)
        
        batch_size = minibatch.obs.shape[0]
        critic_output = jax.tree.map(lambda x: x[:batch_size], joint_output)
        next_output = jax.tree.map(lambda x: x[batch_size:], joint_output)
        
        curr_emb_raw = critic_output["embed"]
        next_emb_raw = next_output["embed"]
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

        # Ξ_B is the empirical replay distribution.
        # Artificial truncation boundaries can be removed, while true terminals remain valid current transitions and only zero the bootstrap term.
        sample_weights = jnp.ones_like(minibatch.done.reshape(-1), dtype=curr_emb_raw.dtype)
        if hparams.mask_truncated:
            sample_weights = sample_weights * (
                1.0 - minibatch.truncated.reshape(-1).astype(curr_emb_raw.dtype)
            )
        sample_weights = sample_weights / jnp.maximum(sample_weights.sum(), 1.0)
        continuation = 1.0 - minibatch.done.reshape(-1).astype(curr_emb_raw.dtype)
        invariance_weights = sample_weights * continuation
        invariance_weights = invariance_weights / jnp.maximum(invariance_weights.sum(), 1.0)

        batch_norm_phi = nnx.merge(train_state.graphdef, batch_norm_phi_params, batch_norm_phi_stats)
        # Normalize with the saved (running) statistics, never the batch statistics.
        batch_norm_phi.eval()
        joint_emb = batch_norm_phi(jnp.concatenate([curr_emb_raw, next_emb_raw, critic_output["pred_features"]], axis=0))
        curr_emb, next_emb, pred_emb = jnp.split(joint_emb, [batch_size, 2 * batch_size], axis=0)
        
        # Update the saved statistics from current/next features only (output discarded).
        batch_norm_phi.train()
        batch_norm_phi(jax.lax.stop_gradient(jnp.concatenate([curr_emb_raw, next_emb_raw], axis=0)))
        batch_norm_phi_stats = nnx.state(batch_norm_phi, nnx.BatchStat)
        
        inv_loss, inv_metrics = invariance_aux_loss(pred_emb, next_emb, invariance_weights)

        gershgorin_eps = float(getattr(hparams, "gershgorin_eps", 1e-5))
        gershgorin_loss_mult = float(
            getattr(hparams, "gershgorin_loss_mult", 1.0)
        )
        if gershgorin_loss_mult != 0.0:
            gershgorin_loss_value, _, gershgorin_metrics = gershgorin_loss(
                curr_emb,
                next_emb,
                sample_weights,
                continuation,
                gamma=hparams.gamma,
                eps=gershgorin_eps,
            )
        else:
            gershgorin_loss_value = jnp.array(0.0, dtype=curr_emb.dtype)
            gershgorin_metrics = {
                "sr_dice/gershgorin_loss": gershgorin_loss_value,
                "sr_dice/gershgorin_margin_min": jnp.array(0.0, dtype=curr_emb.dtype),
                "sr_dice/gershgorin_margin_mean": jnp.array(0.0, dtype=curr_emb.dtype),
            }

        pred_rew = critic_output["pred_rew"]
        value = critic_output["value"]
        aux_rew_loss = optax.squared_error(
            pred_rew.reshape(-1), minibatch.reward.reshape(-1)
        )
        rew_aux_loss = jnp.sum(sample_weights * aux_rew_loss)
        inv_loss_mult = float(getattr(hparams, "inv_loss_mult", 1.0))
        rew_aux_loss_mult = float(getattr(hparams, "rew_aux_loss_mult", 1.0))
        aux_loss = (
            inv_loss_mult * inv_loss
            + gershgorin_loss_mult * gershgorin_loss_value
            + rew_aux_loss_mult * rew_aux_loss
        )

        critic_loss_arr = optax.squared_error(value, target_values)
        critic_loss = jnp.mean(critic_loss_arr)

        mc_error = value.reshape(-1) - target_values.reshape(-1)
        mc_bias = jnp.mean(mc_error)
        mc_error_var = jnp.var(mc_error)

        td_error = (
            minibatch.extras["soft_reward"].reshape(-1)
            + hparams.gamma * minibatch.extras["next_policy_value"].reshape(-1)
            - minibatch.extras["action_value"].reshape(-1)
        )
        td_error_mean = jnp.mean(td_error)
        target_mean = jnp.mean(target_values)
        target_var = jnp.var(target_values)
        q_mean = jnp.mean(value)

        mask = (1.0 - minibatch.truncated) if hparams.mask_truncated else 1.0
        is_w = minibatch.extras.get("is_weight", None)
        per_scale = (
            is_w.reshape(-1)
            if (data_type == "PER" and is_w is not None)
            else 1.0
        )
        td_loss = jnp.mean(per_scale * mask * critic_update_loss)
        td_loss_mult = float(getattr(hparams, "td_loss_mult", 1.0))
        loss = td_loss_mult * td_loss + hparams.aux_loss_mult * aux_loss
        unmasked_critic_total_loss = (
            td_loss_mult * jnp.mean(critic_update_loss)
            + hparams.aux_loss_mult * aux_loss
        )

        critic_diag = {
            "critic_diag/critic_loss": critic_loss,
            "critic_diag/mc_bias": mc_bias,
            "critic_diag/mc_error_var": mc_error_var,
            "critic_diag/td_error_mean": td_error_mean,
            "critic_diag/target_mean": target_mean,
            "critic_diag/target_var": target_var,
            "critic_diag/q": q_mean,
        }


        return loss, ({
            "critic_update_loss": critic_update_loss,
            "masked_critic_total_loss": loss,
            "unmasked_critic_total_loss": unmasked_critic_total_loss,
            "aux_loss": aux_loss,
            "rew_aux_loss": rew_aux_loss,
            "gershgorin_loss": gershgorin_loss_value,
            "abs_batch_action": jnp.abs(minibatch.action).mean(),
            **inv_metrics,
            **gershgorin_metrics,
            **critic_diag,
        }, batch_norm_phi_stats)

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
        actor_objective = actor_loss.reshape(-1)
        if getattr(hparams, "use_sr_dice_ratio", True):
            actor_objective = rho * actor_objective
        dice_actor_loss = actor_objective.mean()

        # Keep KL/entropy unchanged.
        kl_flat = kl.reshape(-1)
        kl_mean = kl_flat.mean()
        entropy_mean = entropy.reshape(-1).mean()
        target_entropy = action_size_target + entropy
        target_entropy_mean = target_entropy.reshape(-1).mean()

        lagrangian = actor_model.lagrangian()
        kl_objective = (
            kl_flat
            * jax.lax.stop_gradient(lagrangian)
            * hparams.reduce_kl
        )
        if hparams.actor_kl_clip_mode == "full":
            loss = jnp.mean(actor_objective + kl_objective)
        elif hparams.actor_kl_clip_mode == "clipped":
            loss = jnp.mean(
                jnp.where(
                    kl_flat < hparams.kl_bound,
                    actor_objective,
                    kl_objective,
                )
            )
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

    train_batch_norm_phi = bool(getattr(hparams, "train_batch_norm_phi", False))

    def critic_update(train_state: REPPOTrainState, minibatch: Transition):
        if train_batch_norm_phi:
            critic_grad_fn = jax.value_and_grad(critic_loss_fn, argnums=(0, 1), has_aux=True)
            output, (critic_grads, batch_norm_phi_grads) = critic_grad_fn(train_state.critic.params, train_state.params["batch_norm_phi"], train_state.params["batch_norm_phi_stats"], train_state, minibatch)
            batch_norm_phi_grads = jax.tree.map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), batch_norm_phi_grads)
        else:
            critic_grad_fn = jax.value_and_grad(critic_loss_fn, argnums=0, has_aux=True)
            output, critic_grads = critic_grad_fn(train_state.critic.params, jax.tree.map(jax.lax.stop_gradient, train_state.params["batch_norm_phi"]), train_state.params["batch_norm_phi_stats"], train_state, minibatch)
        critic_metrics, batch_norm_phi_stats = output[1]
        critic_grads = jax.tree.map(lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), critic_grads)
        train_state = train_state.replace(critic=train_state.critic.apply_gradients(critic_grads))
       
        if train_batch_norm_phi:
            train_state = apply_dice_update(train_state, "batch_norm_phi", batch_norm_phi_grads)
        train_state = train_state.replace(params={**train_state.params, "batch_norm_phi_stats": batch_norm_phi_stats})
        return train_state, critic_metrics

    def actor_update(train_state: REPPOTrainState, minibatch: Transition):
        def update_actor(_):
            actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
            output, grads = actor_grad_fn(train_state.actor.params, train_state, minibatch)
            grads = jax.tree.map(
                lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0),
                grads,
            )
            grad_norm = optax.global_norm(grads)
            return train_state.actor.apply_gradients(grads), grad_norm, output[1]

        def hold_actor(_):
            actor_ref = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
            zero = jnp.array(0.0, dtype=minibatch.obs.dtype)
            lagrangian = actor_ref.lagrangian()
            actor_metrics = {
                "temp": jnp.squeeze(actor_ref.temperature()),
                "abs_pred_action": zero,
                "lagrangian": lagrangian,
                "lagrangian_loss": jnp.zeros_like(lagrangian),
                "entropy_loss": zero,
                "actor_diag/kl": zero,
                "actor_diag/entropy": zero,
                "actor_diag/actor_loss": zero,
                "actor_diag/sr_dice_ratio_mean": jax.lax.stop_gradient(minibatch.extras["sr_dice_ratio"].mean()),
                "actor_diag/real_action_log_prob": zero,
            }
            return train_state.actor, zero, actor_metrics

        if cfg.algorithm.bc_indicator:
            delay = getattr(hparams, "bc_actor_update_delay", 0)
            actor_train_state, grad_norm, actor_metrics = jax.lax.cond(train_state.iteration > delay, update_actor, hold_actor, None)
        else:
            actor_train_state, grad_norm, actor_metrics = update_actor(None)
        return train_state.replace(actor=actor_train_state), {**actor_metrics, "actor_diag/grad_norm": grad_norm}

    def cache_sr_dice_features(key: jax.Array, train_state: REPPOTrainState, batch: Transition, initial_obs: jax.Array):
        """Cache one shared fixed post-critic φ_k for successor TD and SR-DICE."""
        actor_ref = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_ref = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        actor_ref.eval()
        critic_ref.eval()
        next_key, start_key = jax.random.split(key)
        obs = batch.obs.reshape((-1, *batch.obs.shape[2:]))
        next_obs = batch.next_obs.reshape((-1, *batch.next_obs.shape[2:]))
        behavior_action = clip_action_for_critic(batch.action.reshape((-1, *batch.action.shape[2:])), action_space)
        
        # Use the current live policy consistently at replay next states and initial states.
        next_action = sample_actor_action(actor_ref, next_obs, next_key, discrete_actions)
        start_obs = initial_obs.reshape((-1, *batch.obs.shape[2:]))
        start_action = sample_actor_action(actor_ref, start_obs, start_key, discrete_actions)
        
        all_obs = jnp.concatenate([obs, next_obs, start_obs], axis=0)
        all_action = jnp.concatenate([behavior_action, next_action, start_action], axis=0)
        all_phi_raw = jax.lax.stop_gradient(critic_ref(all_obs, all_action)["embed"])
        
        batch_norm_phi = nnx.merge(train_state.graphdef, train_state.params["batch_norm_phi"], train_state.params["batch_norm_phi_stats"])
        # Apply the saved statistics only; they are updated once per critic step in critic_loss_fn.
        batch_norm_phi.eval()
        all_phi = batch_norm_phi(all_phi_raw)
        batch_size = obs.shape[0]
        curr_phi, next_phi, start_phi = jnp.split(all_phi, [batch_size, 2 * batch_size], axis=0)
        curr_phi = curr_phi.reshape((*batch.done.shape, curr_phi.shape[-1]))
        next_phi = next_phi.reshape((*batch.done.shape, next_phi.shape[-1]))
        batch = batch.replace(extras={**batch.extras, "sr_dice_phi": curr_phi, "sr_dice_next_phi": next_phi})
        return train_state, batch, start_phi

    def successor_update(train_state: REPPOTrainState, minibatch: Transition):
        """Update only S_k from cached fixed post-critic features."""
        def successor_loss_fn(successor):
            curr_phi = jax.lax.stop_gradient(minibatch.extras["sr_dice_phi"])
            next_phi = jax.lax.stop_gradient(minibatch.extras["sr_dice_next_phi"])
            weights = jnp.ones_like(minibatch.done.reshape(-1), dtype=curr_phi.dtype)
            if hparams.mask_truncated:
                weights = weights * (1.0 - minibatch.truncated.reshape(-1).astype(curr_phi.dtype))
            weights = weights / jnp.maximum(weights.sum(), 1.0)
            continuation = 1.0 - minibatch.done.reshape(-1).astype(curr_phi.dtype)
            return successor_feature_td_loss(curr_phi, next_phi, weights, continuation, successor, hparams.gamma)

        (successor_loss, successor_metrics), successor_grad = jax.value_and_grad(successor_loss_fn, has_aux=True)(train_state.params["sr_dice_successor"])
        successor_grad = jnp.nan_to_num(successor_grad, nan=0.0, posinf=1.0, neginf=-1.0)
        train_state = apply_dice_update(train_state, "sr_dice_successor", successor_grad)
        return train_state, {"sr_dice/successor_loss": successor_loss}

    def ratio_update(train_state: REPPOTrainState, minibatch: Transition, successor_start_mean: jax.Array):
        """Update only ν_k from cached fixed post-critic features."""
        def ratio_loss_fn(nu):
            dice_params = {**train_state.params, "sr_dice_nu": nu}
            _, ratio_metrics = fit_sr_dice_ratio(dice_params, train_state, minibatch, hparams, action_space, successor_start_mean)
            return ratio_metrics["sr_dice/ratio_loss"], ratio_metrics

        (ratio_loss, ratio_metrics), ratio_grad = jax.value_and_grad(ratio_loss_fn, has_aux=True)(train_state.params["sr_dice_nu"])
        ratio_grad = jnp.nan_to_num(ratio_grad, nan=0.0, posinf=1.0, neginf=-1.0)
        train_state = apply_dice_update(train_state, "sr_dice_nu", ratio_grad)
        return train_state, {
            "sr_dice/ratio_loss": ratio_loss,
            "sr_dice/rho_mean": ratio_metrics["sr_dice/rho_mean"],
            "sr_dice/rho_std": ratio_metrics["sr_dice/rho_std"],
            "sr_dice/rho_ess": ratio_metrics["sr_dice/rho_ess"],
        }

    def run_epoch(
        key: jax.Array, train_state: REPPOTrainState, batch: Transition, update_fn
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        key, shuffle_key, act_key, kl_key = jax.random.split(key, 4)

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
            }
        )

        train_state, metrics = jax.lax.scan(update_fn, train_state, minibatches)

        metrics_mean = jax.tree.map(lambda x: x.mean(0), metrics)
        if "actor_diag/grad_norm" in metrics:
            metrics_mean["actor_diag/grad_norm_var"] = jnp.var(metrics["actor_diag/grad_norm"])
        return train_state, metrics_mean

    def nstep_lambda(batch: Transition):
        # Fast and exact SAC-style one-step target. The replay training path enforces num_steps == 1
        if int(getattr(hparams, "num_steps", 1)) == 1:
            target_value = batch.extras["value"]
            continuation = jnp.where(
                batch.truncated.astype(bool),
                jnp.ones_like(batch.done, dtype=target_value.dtype),
                1.0 - batch.done.astype(target_value.dtype),
            )
            target_values = (
                batch.extras["soft_reward"]
                + hparams.gamma * continuation * target_value
            )
            target_advs = target_values - batch.extras["action_value"]
            retrace_coeff_mean = jnp.array(0.0, dtype=target_values.dtype)
            return target_values, target_advs, retrace_coeff_mean

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
        key, next_key, policy_key, mc_next_key = jax.random.split(key, 4)

        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        target_critic_model = nnx.merge(
            train_state.target_critic.graphdef,
            train_state.target_critic.params,
        )
        actor_model.eval()
        critic_model.eval()
        target_critic_model.eval()

        # SAC one-step target: one reparameterized next action and the matching target-Q/log-prob pair. This is the only target construction needed
        # when num_steps == 1.
        next_pi = actor_model(batch.next_obs)
        next_action, next_log_prob = next_pi.sample_and_log_prob(seed=next_key)
        next_action = clip_action_for_critic(next_action, action_space)
        target_value = target_critic_model(batch.next_obs, next_action)["value"]

        critic_action = clip_action_for_critic(batch.action, action_space)
        action_value = critic_model(batch.obs, critic_action)["value"]

        pi = actor_model(batch.obs)
        current_log_probs = pi.log_prob(critic_action)
        soft_reward = (
            batch.reward
            - hparams.gamma
            * next_log_prob
            * actor_model.temperature()
        )

        if int(getattr(hparams, "num_steps", 1)) == 1:
            return {
                "soft_reward": soft_reward * cfg.env.get("reward_scaling", 1.0),
                "value": target_value,
                "action_value": action_value,
                # These aliases keep diagnostics compatible without performing the old 8-action Monte-Carlo critic evaluations.
                "policy_value": action_value,
                "next_policy_value": target_value,
                "target_next_policy_value": target_value,
                "next_action": next_action,
                "log_prob": current_log_probs,
                "current_log_prob": current_log_probs,
            }

        # Multi-step fallback retained for non-replay/on-policy configurations.
        if hparams.scale_samples_with_action_d:
            num_samples = 8 * d
        else:
            num_samples = 8

        next_actions = next_pi.sample(seed=mc_next_key, sample_shape=(num_samples,))
        next_actions = clip_action_for_critic(next_actions, action_space)
        next_obs_tiled = jnp.repeat(
            batch.next_obs[None, ...], next_actions.shape[0], axis=0
        )
        next_policy_value = critic_model(next_obs_tiled, next_actions)["value"].mean(0)
        target_next_policy_value = target_critic_model(
            next_obs_tiled, next_actions
        )["value"].mean(0)

        policy_actions = pi.sample(seed=policy_key, sample_shape=(num_samples,))
        policy_actions = clip_action_for_critic(policy_actions, action_space)
        obs_tiled = jnp.repeat(batch.obs[None, ...], policy_actions.shape[0], axis=0)
        policy_value = critic_model(obs_tiled, policy_actions)["value"].mean(0)

        return {
            "soft_reward": soft_reward * cfg.env.get("reward_scaling", 1.0),
            "value": target_value,
            "action_value": action_value,
            "policy_value": policy_value,
            "next_policy_value": next_policy_value,
            "target_next_policy_value": target_next_policy_value,
            "next_action": next_action,
            "log_prob": current_log_probs,
            "current_log_prob": current_log_probs,
        }

    def learner_fn(
        key: Key, train_state: REPPOTrainState, batch: Transition
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        # Snapshot the live actor here so this epoch is the local update pi_k -> pi_{k+1}, with KL measured against pi_k.

        # SR-DICE is always fitted and logged. `use_sr_dice_ratio` controls only whether the detached fitted ratio weights the actor objective.
        if "initial_obs" not in batch.extras:
            raise KeyError("SR-DICE diagnostics require a separately sampled batch.extras['initial_obs'] with s₀ ∼ d₀.")
        initial_obs = batch.extras.get("initial_obs", None)
        batch = batch.replace(extras={k: v for k, v in batch.extras.items() if k != "initial_obs"})
        if hparams.normalize_env:
            new_norm_state = normalizer.update(train_state.normalization_state, batch.obs)
            if initial_obs is not None:
                initial_obs = normalizer.normalize(train_state.normalization_state, initial_obs)
            batch = batch.replace(obs=normalizer.normalize(train_state.normalization_state, batch.obs), next_obs=normalizer.normalize(train_state.normalization_state, batch.next_obs))
            train_state = train_state.replace(normalization_state=new_norm_state)

        key, diag_key = jax.random.split(key)

        diagnostics_extras = compute_extras(key=diag_key, train_state=train_state, batch=batch)
        batch = batch.replace(extras={**batch.extras, **diagnostics_extras})

        target_values, target_advs, retrace_coeff_mean = nstep_lambda(batch)
        batch = batch.replace(extras={**batch.extras, "target_values": target_values, "target_advs": target_advs})

        current_log_prob = batch.extras["current_log_prob"]
        reward_mean = batch.reward.mean()
        policy_log_prob_mean = current_log_prob.mean()
        per_env_td_error = jnp.abs(batch.extras["action_value"] - batch.extras["target_values"]).mean(axis=0)
        key, critic_key, cache_key, successor_key, sr_dice_key, pv_before_key, actor_key, pv_after_key = jax.random.split(key, 8)

        # Critic/representation phase: the live actor is fixed until the later actor update.
        # Invariance and Gershgorin therefore use one consistent current policy in this learner call.
        train_state, critic_metrics = jax.lax.scan(
            lambda state, epoch_key: run_epoch(
                epoch_key, state, batch, critic_update
            ),
            train_state,
            jax.random.split(critic_key, int(getattr(hparams, "num_critic_epochs", 2))),
        )
        critic_metrics = jax.tree.map(lambda x: x[-1], critic_metrics)
        train_state = polyak_update_target_critic(train_state)

        # Cache the final fixed φ_k once; S_k, ν_k, and ρ reuse these tensors without network forwards.
        train_state, batch, start_phi = cache_sr_dice_features(
            cache_key, train_state, batch, initial_obs
        )

        # Fit the separate successor-feature matrix S_k by semi-gradient TD.
        train_state, successor_metrics = jax.lax.scan(
            lambda state, epoch_key: run_epoch(
                epoch_key, state, batch, successor_update
            ),
            train_state,
            jax.random.split(successor_key, 1),
        )
        successor_metrics = jax.tree.map(lambda x: x[-1], successor_metrics)

        # The SR-DICE anchor uses live-policy initial features and the newly fitted successor matrix.
        successor_start_mean = compute_successor_start_mean(train_state.params, start_phi)

        # Fit ν with the successor matrix and cached features fixed.
        train_state, ratio_metrics = jax.lax.scan(
            lambda state, epoch_key: run_epoch(
                epoch_key, state, batch, lambda s, mb: ratio_update(s, mb, successor_start_mean),
            ),
            train_state,
            jax.random.split(sr_dice_key, 1),
        )
        ratio_metrics = jax.tree.map(lambda x: x[-1], ratio_metrics)

        # Re-evaluate the fitted ratio after the ν update so all logged ratio
        # diagnostics correspond to the parameters actually passed to the actor.
        sr_dice_ratio, fitted_ratio_metrics = fit_sr_dice_ratio(
            train_state.params,
            train_state,
            batch,
            hparams,
            action_space,
            successor_start_mean,
        )
        dice_metrics = {
            "sr_dice/successor_loss": successor_metrics["sr_dice/successor_loss"],
            "sr_dice/ratio_loss": fitted_ratio_metrics["sr_dice/ratio_loss"],
            "sr_dice/rho_mean": fitted_ratio_metrics["sr_dice/rho_mean"],
            "sr_dice/rho_std": fitted_ratio_metrics["sr_dice/rho_std"],
            "sr_dice/rho_ess": fitted_ratio_metrics["sr_dice/rho_ess"],
        }
        batch = batch.replace(
            extras={
                **batch.extras,
                "sr_dice_ratio": jax.lax.stop_gradient(sr_dice_ratio),
            }
        )

        # Optional policy-improvement diagnostic. Disabled by default because it adds two full actor/critic passes.
        actor_before = nnx.merge(train_state.actor.graphdef, train_state.actor_target.params)
        critic_before = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        actor_before.eval()
        critic_before.eval()
        action_before = clip_action_for_critic(actor_before(batch.obs).sample(seed=pv_before_key), action_space)
        policy_value_before_vec = critic_before(batch.obs, action_before)["value"].reshape(-1)

        # Actor-only policy-improvement phase with fixed critic and detached fitted ratio.
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
        action_after = clip_action_for_critic(actor_after(batch.obs).sample(seed=pv_after_key), action_space)
        policy_value_after_vec = critic_after(batch.obs, action_after)["value"].reshape(-1)
        policy_improvement = jnp.mean(
            policy_value_after_vec - policy_value_before_vec
        )

        base_metrics = {
            "reward_mean": reward_mean,
            "actor_diag/policy_log_prob_mean": policy_log_prob_mean,
            "critic_diag/retrace_coeff_mean": retrace_coeff_mean,
            "actor_diag/retrace_coeff_mean": retrace_coeff_mean,
            "actor_diag/policy_improvement": policy_improvement,
            "sys/grad_updates": (
                train_state.time_steps
                // (hparams.num_steps * hparams.num_envs)
            )
            * 2
            * int(getattr(hparams, "num_epochs", 1))
            * hparams.num_mini_batches,
        }
        return train_state, {**critic_metrics, **actor_metrics, **dice_metrics, **base_metrics}, per_env_td_error

    return jax.jit(learner_fn)