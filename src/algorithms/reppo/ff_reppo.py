import logging
import math
from typing import Callable
import operator
import os

import hydra
import jax
import optax
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
import distrax

logging.basicConfig(level=logging.INFO)

def batch_orthonormality_loss(
    features: jax.Array,
    weights: jax.Array,
    reward: jax.Array,
    successor: jax.Array,
):
    weights = jax.lax.stop_gradient(weights)
    feature_dim = features.shape[-1]
    identity = jnp.eye(feature_dim, dtype=features.dtype)
    gram = features.T @ (weights[:, None] * features)
    loss = 0.5 * jnp.sum(jnp.square(gram - identity)) / gram.size

    # Logging only: mean absolute off-diagonal entry of the normalized Gram matrix.
    gram_metrics = jax.lax.stop_gradient(gram)
    features_metrics = jax.lax.stop_gradient(features)
    reward_metrics = jax.lax.stop_gradient(reward.reshape(-1).astype(features.dtype))
    feature_scale = jnp.sqrt(jnp.clip(jnp.diag(gram_metrics), a_min=1e-8))
    correlation = gram_metrics / (
        feature_scale[:, None] * feature_scale[None, :] + 1e-8
    )
    off_diagonal_sum = (
        jnp.sum(jnp.abs(correlation))
        - jnp.sum(jnp.abs(jnp.diag(correlation)))
    )
    feature_correlation = off_diagonal_sum / max(feature_dim * (feature_dim - 1), 1)

    gram_eigenvalues, gram_eigenvectors = jnp.linalg.eigh(gram_metrics)
    gram_min_eigenval = gram_eigenvalues[0]
    gram_max_eigenval = gram_eigenvalues[-1]

    # Reward energy along each Gram eigendirection:
    # E_i = ((Phi v_i)^T Xi r)^2 / lambda_i.
    feature_reward = features_metrics.T @ (weights * reward_metrics)
    reward_alignment = gram_eigenvectors.T @ feature_reward
    reward_energy = jnp.where(
        gram_eigenvalues > 1e-8,
        jnp.square(reward_alignment) / jnp.clip(gram_eigenvalues, a_min=1e-8),
        0.0,
    )

    # Logging only: conditioning of the immediate-reward-relevant subspace and its successor-propagated future-value subspace.
    # U_r contains the Gram eigendirections with the largest reward energy.
    # With row-wise successor features Psi = Phi S and r ~= Phi w,
    # Q^pi ~= Psi w = Phi S w, so span(S U_r) is the corresponding successor-propagated value-relevant subspace.
    subspace_dim = min(10, feature_dim)
    _, reward_indices = jax.lax.top_k(reward_energy, subspace_dim)
    reward_basis = gram_eigenvectors[:, reward_indices]

    reward_subspace_gram = reward_basis.T @ gram_metrics @ reward_basis
    reward_subspace_eigenvalues = jnp.linalg.eigvalsh(reward_subspace_gram)
    reward_subspace_kappa = (
        jnp.clip(reward_subspace_eigenvalues[-1], a_min=0.0)
        / jnp.clip(reward_subspace_eigenvalues[0], a_min=1e-8)
    )

    successor_metrics = jax.lax.stop_gradient(successor)
    value_basis_raw = successor_metrics @ reward_basis
    value_basis, _ = jnp.linalg.qr(value_basis_raw, mode="reduced")

    value_subspace_gram = value_basis.T @ gram_metrics @ value_basis
    value_subspace_eigenvalues = jnp.linalg.eigvalsh(value_subspace_gram)
    value_subspace_kappa = (
        jnp.clip(value_subspace_eigenvalues[-1], a_min=0.0)
        / jnp.clip(value_subspace_eigenvalues[0], a_min=1e-8)
    )

    k = min(10, feature_dim)
    bottom10_reward_energy = reward_energy[:k]
    top10_reward_energy = reward_energy[-k:]

    return loss, gram, {
        "gram_min_eigenval": gram_min_eigenval,
        "gram_max_eigenval": gram_max_eigenval,
        "feature_correlation": feature_correlation,
        "reward_energy_top10_mean": top10_reward_energy.mean(),
        "reward_energy_top10_var": top10_reward_energy.var(),
        "reward_energy_bottom10_mean": bottom10_reward_energy.mean(),
        "reward_energy_bottom10_var": bottom10_reward_energy.var(),
        "reward_subspace_kappa": reward_subspace_kappa,
        "value_subspace_kappa": value_subspace_kappa,
    }

def gershgorin_loss(curr_features: jax.Array, next_features: jax.Array, weights: jax.Array, continuation: jax.Array, gamma: float, eps: float = 1e-5):
    """Gershgorin loss on the empirical TD iteration matrix.

    With row-wise features Φ_B ∈ R^{B×d}, the sampled matrix is

        A_B = Φ_Bᵀ Ξ_B (Φ_B - γ P^πΦ_B)
            ≈ φ(s,a)ᵀ Ξ_B [φ(s,a) - γφ(s',a')].

    Gradients flow through the current features only; policy-next features are
    treated as a fixed semi-gradient target.
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

    # Compute the minimum real eigenvalue for logging
    min_real_eigenval = jnp.min(jnp.real(jnp.linalg.eig(jax.lax.stop_gradient(td_matrix))[0]))

    return loss, td_matrix, {
        "gershgorin_margin_min": margin.min(),
        "gershgorin_margin_mean": margin.mean(),
        "Aphi_min_real_eigenval": min_real_eigenval,
    }


def successor_feature_td_loss(curr_features: jax.Array, next_features: jax.Array, weights: jax.Array, continuation: jax.Array, successor: jax.Array, gamma: float):
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
    return jnp.sum(weights * per_sample_loss) / curr_features.shape[-1]


def fit_sr_dice_ratio(nu: jax.Array, features: jax.Array, weights: jax.Array, successor_start_mean: jax.Array, gamma: float):
    features = jax.lax.stop_gradient(features)
    weights = jax.lax.stop_gradient(weights)
    successor_start_mean = jax.lax.stop_gradient(successor_start_mean)
    rho = features @ nu
    ratio_loss = 0.5 * jnp.sum(weights * jnp.square(rho)) - (1.0 - gamma) * jnp.dot(nu, successor_start_mean)
    valid_mask = weights > 0.0
    rho_ess_weights = jnp.where(valid_mask, jnp.clip(rho, a_min=1e-5), jnp.zeros_like(rho))
    rho_ess = (jnp.sum(rho_ess_weights) ** 2) / (
        jnp.sum(jnp.square(rho_ess_weights)) + 1e-8
    )
    return ratio_loss, {
        "sr_dice/ratio_loss": ratio_loss,
        "sr_dice/rho_mean": rho.mean(),
        "sr_dice/rho_std": rho.std(),
        "sr_dice/rho_ess": rho_ess,
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


def make_init_fn(
    cfg: DictConfig,
    observation_space: Space,
    action_space: Space,
) -> InitFn:
    hparams = cfg.algorithm

    def init(key: Key):
        key, model_key = jax.random.split(key)
        rngs = nnx.Rngs(model_key)

        optim_fn = hydra.utils.instantiate(hparams.optimizer)

        if hparams.max_grad_norm is not None:
            tx = optax.chain(optax.clip_by_global_norm(hparams.max_grad_norm), optim_fn)
        else:
            tx = optim_fn

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

        dummy_obs = jax.tree.map(
            lambda x: jnp.zeros((1,) + x.shape, dtype=jnp.float32),
            observation_space.sample(key),
        )
        if isinstance(action_space, Discrete):
            dummy_action = jnp.zeros((1,), dtype=jnp.int32)
        else:
            dummy_action = jnp.zeros((1,) + action_space.shape, dtype=jnp.float32)
        feature_dim = critic(dummy_obs, dummy_action)["embed"].shape[-1]

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

        return REPPOTrainState.create(
            graphdef=nnx.graphdef(actor),
            params={
                "sr_dice_nu": jnp.zeros((feature_dim,), dtype=jnp.float32),
                "sr_dice_successor": jnp.eye(feature_dim, dtype=jnp.float32),
                "sr_dice_start_phi": jnp.zeros((hparams.num_envs, feature_dim), dtype=jnp.float32),
            },
            tx=optax.set_to_zero(),
            actor=nnx.TrainState.create(
                graphdef=nnx.graphdef(actor), params=nnx.state(actor), tx=tx
            ),
            critic=nnx.TrainState.create(
                graphdef=nnx.graphdef(critic), params=nnx.state(critic), tx=tx
            ),
            target_critic=nnx.TrainState.create(
                graphdef=nnx.graphdef(critic),
                params=nnx.state(critic),
                tx=optax.set_to_zero(),
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

    def critic_loss_fn(
        params: nnx.Param, train_state: REPPOTrainState, minibatch: Transition
    ):
        critic_model = nnx.merge(train_state.critic.graphdef, params)
        critic_model.train()

        # The critic embed is the single canonical raw Phi used by Q, invariance, and Gershgorin.
        critic_output = critic_model(minibatch.obs, minibatch.action)
        curr_features = critic_output["embed"]
        value = critic_output["value"]
        pred_features = critic_output["pred_features"]
        pred_rew = critic_output["pred_rew"]

        target_values = minibatch.extras["target_values"]

        if hparams.hl_gauss:
            target_cat = jax.vmap(utils.hl_gauss, in_axes=(0, None, None, None))(
                target_values, hparams.num_bins, hparams.vmin, hparams.vmax
            )
            critic_pred = critic_output["logits"]
            critic_update_loss = optax.softmax_cross_entropy(critic_pred, target_cat)
        else:
            critic_update_loss = optax.squared_error(
                value.reshape(-1, 1),
                target_values.reshape(-1, 1),
            )

        # Invariance is defined in the same raw Phi space used by Q and Gershgorin.
        # The policy-next feature remains a semi-gradient target.
        done_mask = 1.0 - minibatch.done.reshape(-1, 1).astype(pred_features.dtype)
        invariance_error = optax.squared_error(
            pred_features,
            minibatch.extras["next_emb"],
        )
        reward_error = optax.squared_error(
            pred_rew,
            minibatch.reward.reshape(-1, 1),
        )
        # Preserve the exact old concatenated-loss weighting while keeping invariance and reward prediction separate for ablations.
        feature_dim = pred_features.shape[-1]
        inv_loss_mult = (float(getattr(hparams, "inv_loss_mult", 0.0)) * feature_dim / (feature_dim + 1))
        rew_aux_loss_mult = (float(getattr(hparams, "rew_aux_loss_mult", 0.0)) / (feature_dim + 1))

        invariance_loss = jnp.mean(done_mask * invariance_error, axis=-1)
        aux_rew_loss = jnp.mean(done_mask * reward_error, axis=-1)
        logged_aux_loss = jnp.mean(done_mask * jnp.concatenate([invariance_error, reward_error], axis=-1), axis=-1)
        logged_invariance_loss = jnp.mean(done_mask * invariance_error)

        # total_aux_loss = jnp.mean(
        #     (1 - minibatch.done.reshape(-1, 1))
        #     * jnp.concatenate([invariance_loss, aux_rew_loss], axis=-1),
        #     axis=-1,
        # )

        # compute l2 error for logging
        critic_loss_arr = optax.squared_error(
            value,
            target_values,
        )
        critic_loss = jnp.mean(critic_loss_arr)

        # Critic bias and error variance (using n-step/Retrace targets as G_t)
        mc_error = value.reshape(-1) - target_values.reshape(-1)
        mc_bias = jnp.mean(mc_error)
        mc_error_var = jnp.var(mc_error)

        # TD error: r + γ·E_a'[Q(s',a')] - Q(s,a)
        # td_error = (
        #     minibatch.extras["soft_reward"].reshape(-1)
        #     + hparams.gamma * minibatch.extras["next_policy_value"].reshape(-1)
        #     - minibatch.extras["action_value"].reshape(-1)
        # )
        # for one step TD
        td_error = target_values.reshape(-1) - value.reshape(-1)
        td_error_mean = jnp.mean(td_error)
        target_mean = jnp.mean(target_values)
        target_var = jnp.var(target_values)
        q_mean = jnp.mean(value)

        mask_truncated = hparams.mask_truncated
        mask = (1.0 - minibatch.truncated) if mask_truncated else 1.0
        # PER: scale loss by importance-sampling weight to correct for sampling bias
        is_w = minibatch.extras.get("is_weight", None)
        per_scale = is_w.reshape(-1) if (data_type == 'PER' and is_w is not None) else 1.0

        # Always compute the requested feature diagnostics. Only use their losses when use_added_loss=True.
        next_features = minibatch.extras["diagnostic_next_emb"]
        sample_weights = jnp.ones_like(
            minibatch.done.reshape(-1), dtype=curr_features.dtype
        )
        if hparams.mask_truncated:
            sample_weights = sample_weights * (
                1.0 - minibatch.truncated.reshape(-1).astype(curr_features.dtype)
            )
        sample_weights = sample_weights / jnp.maximum(sample_weights.sum(), 1.0)
        continuation = jnp.where(
            minibatch.truncated.reshape(-1).astype(bool),
            jnp.ones_like(minibatch.done.reshape(-1), dtype=curr_features.dtype),
            1.0 - minibatch.done.reshape(-1).astype(curr_features.dtype),
        )

        computed_gershgorin_loss, _, gershgorin_metrics = gershgorin_loss(
            curr_features,
            next_features,
            sample_weights,
            continuation,
            gamma=hparams.gamma,
            eps=float(getattr(hparams, "gershgorin_eps", 1e-5)),
        )
        computed_orth_loss, _, orth_metrics = batch_orthonormality_loss(
            curr_features,
            sample_weights,
            minibatch.reward,
            train_state.params["sr_dice_successor"],
        )
        use_added_loss = bool(getattr(hparams, "use_added_loss", False))
        gershgorin_loss_value = jnp.where(
            use_added_loss,
            computed_gershgorin_loss,
            jnp.array(0.0, dtype=value.dtype),
        )
        orth_loss_value = jnp.where(
            use_added_loss,
            computed_orth_loss,
            jnp.array(0.0, dtype=value.dtype),
        )

        critic_objective = (
            critic_update_loss
            + hparams.aux_loss_mult
            * inv_loss_mult * invariance_loss + rew_aux_loss_mult * aux_rew_loss
        )

        unmasked_critic_total_loss = jnp.mean(critic_objective)
        loss = jnp.mean(per_scale * mask * critic_objective)

        # Gershgorin and orthonormality are batch-level matrix losses, so add them after reducing the per-sample critic objective.
        if float(getattr(hparams, "gershgorin_loss_mult", 0.0)) > 0.0:
            weighted_gershgorin_loss = float(getattr(hparams, "gershgorin_loss_mult", 0.0)) * gershgorin_loss_value
            unmasked_critic_total_loss += weighted_gershgorin_loss
            loss += weighted_gershgorin_loss

        if float(getattr(hparams, "orth_loss_mult", 0.0)) > 0.0:
            weighted_orth_loss = float(getattr(hparams, "orth_loss_mult", 0.0)) * orth_loss_value
            unmasked_critic_total_loss += weighted_orth_loss
            loss += weighted_orth_loss

        return loss, dict(
            value_loss=critic_loss,
            critic_update_loss=critic_update_loss,
            loss=loss,
            aux_loss=logged_aux_loss,
            invariance_loss=logged_invariance_loss,
            rew_aux_loss=reward_error,
            **orth_metrics,
            **gershgorin_metrics,
            q=value.mean(),
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            reward_mean=minibatch.reward.mean(),
            target_values=target_values.mean(),
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

        # set up models for training
        actor_model.train()
        critic_target_model.eval()
        actor_target_model.eval()
        pi = actor_model(minibatch.obs)
        old_pi = actor_target_model(minibatch.obs)

        # policy KL constraint
        kl = compute_policy_kl(minibatch=minibatch, pi=pi, old_pi=old_pi)
        alpha = jax.lax.stop_gradient(actor_model.temperature())
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
                q_values = critic_pred["value"]
                value = q_values.sum(axis=0, keepdims=True)
                value = (value - q_values) / (q_values.shape[0] - 1)
                adv = q_values - value
                actor_loss = -jnp.mean(log_prob * jax.lax.stop_gradient(adv) - alpha * log_prob, axis=0)
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

        lagrangian = actor_model.lagrangian()

        if hparams.actor_kl_clip_mode == "full":
            loss = jnp.mean(
                actor_loss + kl * jax.lax.stop_gradient(lagrangian) * hparams.reduce_kl
            )
        elif hparams.actor_kl_clip_mode == "clipped":
            loss = jnp.mean(
                jnp.where(
                    kl < hparams.kl_bound,
                    actor_loss,
                    kl * jax.lax.stop_gradient(lagrangian) * hparams.reduce_kl,
                )
            )
        elif hparams.actor_kl_clip_mode == "value":
            loss = jnp.mean(actor_loss)
        else:
            raise ValueError(f"Unknown actor loss mode: {hparams.actor_kl_clip_mode}")

        # SAC target entropy loss
        target_entropy = action_size_target + entropy
        target_entropy_loss = actor_model.temperature() * jax.lax.stop_gradient(target_entropy)

        # Lagrangian constraint (follows temperature update)
        lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl - hparams.kl_bound)

        # total loss
        if hparams.update_entropy_lagrangian:
            loss += jnp.mean(target_entropy_loss)
        if hparams.update_kl_lagrangian:
            loss += jnp.mean(lagrangian_loss)

        # for logging
        real_action_log_prob = old_pi.log_prob(minibatch.action.clip(-0.999, 0.999)).mean()

        return loss, dict(
            actor_loss=actor_loss,
            loss=loss,
            temp=actor_model.temperature(),
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            abs_pred_action=jnp.abs(pred_action).mean()
            if not discrete_actions
            else 0.0,
            reward_mean=minibatch.reward.mean(),
            kl=kl.mean(),
            lagrangian=lagrangian,
            lagrangian_loss=lagrangian_loss,
            entropy=entropy,
            entropy_loss=target_entropy_loss,
            target_values=minibatch.extras["target_values"].mean(),
            real_action_log_prob=real_action_log_prob,
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

    def compute_epoch_diagnostics(key: Key, train_state: REPPOTrainState, batch: Transition):
        next_action_key, _ = jax.random.split(key)
        actor_target_model = nnx.merge(train_state.actor.graphdef, train_state.actor_target.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        actor_target_model.eval()
        critic_model.eval()
        next_action = actor_target_model(batch.next_obs).sample(seed=next_action_key)
        if not discrete_actions:
            next_action = next_action.clip(-0.999, 0.999)
        next_features = jax.lax.stop_gradient(critic_model(batch.next_obs, next_action)["embed"])
        return batch.replace(extras={**batch.extras, "diagnostic_next_emb": next_features})

    def compute_dice_features(key: Key, train_state: REPPOTrainState, batch: Transition, initial_obs: jax.Array):
        next_action_key, start_action_key = jax.random.split(key)
        actor_target_model = nnx.merge(train_state.actor.graphdef, train_state.actor_target.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        actor_target_model.eval()
        critic_model.eval()
        next_action = actor_target_model(batch.next_obs).sample(seed=next_action_key)
        start_action = actor_target_model(initial_obs).sample(seed=start_action_key)
        if not discrete_actions:
            next_action = next_action.clip(-0.999, 0.999)
            start_action = start_action.clip(-0.999, 0.999)
        curr_features = jax.lax.stop_gradient(critic_model(batch.obs, batch.action)["embed"])
        next_features = jax.lax.stop_gradient(critic_model(batch.next_obs, next_action)["embed"])
        start_features = jax.lax.stop_gradient(critic_model(initial_obs, start_action)["embed"])
        return batch.replace(extras={**batch.extras, "sr_dice_phi": curr_features, "sr_dice_next_phi": next_features}), start_features

    def dice_update(train_state: REPPOTrainState, minibatch: Transition):
        curr_features = jax.lax.stop_gradient(minibatch.extras["sr_dice_phi"])
        next_features = jax.lax.stop_gradient(minibatch.extras["sr_dice_next_phi"])
        start_features = jax.lax.stop_gradient(train_state.params["sr_dice_start_phi"])

        weights = jnp.ones_like(minibatch.done.reshape(-1), dtype=curr_features.dtype)
        if hparams.mask_truncated:
            weights = weights * (
                1.0 - minibatch.truncated.reshape(-1).astype(curr_features.dtype)
            )
        weights = weights / jnp.maximum(weights.sum(), 1.0)
        continuation = jnp.where(
            minibatch.truncated.reshape(-1).astype(bool),
            jnp.ones_like(minibatch.done.reshape(-1), dtype=curr_features.dtype),
            1.0 - minibatch.done.reshape(-1).astype(curr_features.dtype),
        )
        dice_lr = float(getattr(hparams, "sr_dice_lr", 1e-5))

        def successor_loss_fn(successor):
            return successor_feature_td_loss(
                curr_features,
                next_features,
                weights,
                continuation,
                successor,
                hparams.gamma,
            )

        successor_loss, successor_grad = jax.value_and_grad(successor_loss_fn)(train_state.params["sr_dice_successor"])
        successor_grad = jnp.nan_to_num(successor_grad, nan=0.0, posinf=1.0, neginf=-1.0)
        successor = train_state.params["sr_dice_successor"] - dice_lr * successor_grad
        successor_start_mean = jax.lax.stop_gradient((start_features @ jax.lax.stop_gradient(successor)).mean(axis=0))

        def ratio_loss_fn(nu):
            return fit_sr_dice_ratio(
                nu,
                curr_features,
                weights,
                successor_start_mean,
                hparams.gamma,
            )

        (_, _), nu_grad = jax.value_and_grad(ratio_loss_fn, has_aux=True)(train_state.params["sr_dice_nu"])
        nu_grad = jnp.nan_to_num(nu_grad, nan=0.0, posinf=1.0, neginf=-1.0)
        nu = train_state.params["sr_dice_nu"] - dice_lr * nu_grad
        ratio_loss, ratio_metrics = fit_sr_dice_ratio(nu, curr_features, weights, successor_start_mean, hparams.gamma)
        train_state = train_state.replace(
            params={
                **train_state.params,
                "sr_dice_successor": successor,
                "sr_dice_nu": nu,
            }
        )
        return train_state, {
            **ratio_metrics,
            "sr_dice/successor_loss": successor_loss,
            "sr_dice/ratio_loss": ratio_loss,
        }

    def update(train_state: REPPOTrainState, batch: Transition):
        # Update critic always
        if cfg.algorithm.bc_indicator:
            def update_critic(_):
                critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                output, grads = critic_grad_fn(train_state.critic.params, train_state, batch)
                critic_train_state = train_state.critic.apply_gradients(grads)
                critic_metrics = output[1]
                return critic_train_state, critic_metrics

            critic_train_state, critic_metrics = update_critic(None)
            train_state = train_state.replace(
                critic=critic_train_state,
            )

            # Update the one-step TD target critic after every critic optimizer step.
            polyak = float(getattr(hparams, "polyak", 0.005))
            target_critic_params = jax.tree.map(
                lambda target, live: (1.0 - polyak) * target + polyak * live,
                train_state.target_critic.params,
                train_state.critic.params,
            )
            train_state = train_state.replace(
                target_critic=train_state.target_critic.replace(params=target_critic_params)
            )

            delay = cfg.algorithm.bc_actor_update_delay

            def update_actor(_):
                actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
                output, grads = actor_grad_fn(train_state.actor.params, train_state, batch)
                grad_norm = jax.tree.map(lambda x: jnp.linalg.norm(x), grads)
                grad_norm = jax.tree.reduce(operator.add, grad_norm)
                grads = jax.tree.map(
                    lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), grads
                )
                actor_train_state = train_state.actor.apply_gradients(grads)
                actor_metrics = output[1]
                return actor_train_state, grad_norm, actor_metrics
            
            def hold_update_actor(_):
                actor_train_state = train_state.actor
                grad_norm = jnp.array(0.0)
                # Dynamically get the metric structure from actor_loss without computing gradients
                _, actor_metrics = actor_loss(train_state.actor.params, train_state, batch)
                return actor_train_state, grad_norm, actor_metrics
            
            # for bc policy only, start updating the actor after some iterations
            actor_train_state, grad_norm, actor_metrics = jax.lax.cond(
                train_state.iteration > delay,
                update_actor,
                hold_update_actor,
                None
            )

        else:
            critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
            output, grads = critic_grad_fn(train_state.critic.params, train_state, batch)
            critic_metrics = output[1]
            critic_train_state = train_state.critic.apply_gradients(grads)
            train_state = train_state.replace(
                critic=critic_train_state,
            )

            # Update the one-step TD target critic after every critic optimizer step.
            polyak = float(getattr(hparams, "polyak", 0.005))
            target_critic_params = jax.tree.map(
                lambda target, live: (1.0 - polyak) * target + polyak * live,
                train_state.target_critic.params,
                train_state.critic.params,
            )
            train_state = train_state.replace(
                target_critic=train_state.target_critic.replace(params=target_critic_params)
            )

            actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
            output, grads = actor_grad_fn(train_state.actor.params, train_state, batch)
            grad_norm = jax.tree.map(lambda x: jnp.linalg.norm(x), grads)
            grad_norm = jax.tree.reduce(operator.add, grad_norm)
            grads = jax.tree.map(
                lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), grads
            )
            actor_train_state = train_state.actor.apply_gradients(grads)
            actor_metrics = output[1]
        
        # Update train_state with actor in all paths
        train_state = train_state.replace(
            actor=actor_train_state,
        )

        return train_state, {
            **critic_metrics,
            **actor_metrics,
            "grad_norm": grad_norm,
        }

    def run_epoch(
        key: jax.Array, train_state: REPPOTrainState, batch: Transition, initial_obs: jax.Array
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:

        if getattr(hparams, "use_one_step_td", True):
            key, target_key = jax.random.split(key)
            extras = compute_extras(
                key=target_key,
                train_state=train_state,
                batch=batch,
            )
            batch.extras.update(extras)
            (
                batch.extras["target_values"],
                batch.extras["target_advs"],
                _,
            ) = nstep_lambda(batch=batch)

        diagnostic_key = jax.random.fold_in(key, 2718)
        dice_key = jax.random.fold_in(key, 31415)
        batch = compute_epoch_diagnostics(diagnostic_key, train_state, batch)

        # Shuffle data and split into mini-batches
        key, shuffle_key, act_key, kl_key = jax.random.split(key, 4)
        batch_size = batch.obs.shape[0]
        mini_batch_size = batch_size // hparams.num_mini_batches
        indices = jax.random.permutation(
            shuffle_key, batch_size
        )
        minibatch_idxs = jax.tree.map(
            lambda x: x.reshape(
                (hparams.num_mini_batches, mini_batch_size, *x.shape[1:])
            ),
            indices,
        )
        minibatches = jax.tree.map(lambda x: jnp.take(x, minibatch_idxs, axis=0), batch)
        minibatches.extras["action_key"] = jax.random.split(
            act_key, hparams.num_mini_batches
        )
        minibatches.extras["kl_key"] = jax.random.split(
            kl_key, hparams.num_mini_batches
        )

        # Run model update for each mini-batch
        train_state, metrics = jax.lax.scan(update, train_state, minibatches)

        # Compute mean metrics across mini-batches
        metrics_mean = jax.tree.map(lambda x: x.mean(0), metrics)

        dice_feature_key, dice_shuffle_key = jax.random.split(dice_key)
        dice_batch, start_features = compute_dice_features(dice_feature_key, train_state, batch, initial_obs)
        train_state = train_state.replace(params={**train_state.params, "sr_dice_start_phi": start_features})
        dice_indices = jax.random.permutation(dice_shuffle_key, batch_size)
        dice_minibatch_idxs = jax.tree.map(
            lambda x: x.reshape(
                (hparams.num_mini_batches, mini_batch_size, *x.shape[1:])
            ),
            dice_indices,
        )
        dice_minibatches = jax.tree.map(lambda x: jnp.take(x, dice_minibatch_idxs, axis=0), dice_batch)
        train_state, dice_metrics = jax.lax.scan(dice_update, train_state, dice_minibatches)
        dice_metrics_mean = jax.tree.map(lambda x: x.mean(0), dice_metrics)
        # Compute max metrics across mini-batches
        # metrics_max = jax.tree.map(lambda x: x.max(), metrics)
        # metrics_min = jax.tree.map(lambda x: x.min(), metrics)
        return (
            train_state,
            {**metrics_mean, **dice_metrics_mean},
        )  # {**metrics_mean, **{k + "_max": v for k, v in metrics_max.items()}, **{k + "_min": v for k, v in metrics_min.items()}}

    def nstep_lambda(batch: Transition):
        if not getattr(hparams, "use_one_step_td", True):
            def loop(carry: tuple[jax.Array, ...], transition: Transition):
                lambda_return, gae, truncated, next_value = carry

                # combine importance_weights with TD lambda
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

                truncated = transition.truncated
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
            # One-step TD target
            target_values = (
                batch.extras["soft_reward"]
                + hparams.gamma * jnp.where(
                    batch.truncated, batch.extras["value"], (1.0 - batch.done.astype(jnp.float32)) * batch.extras["value"],
                )
            )

            target_advs = target_values - batch.extras["action_value"]
            retrace_coeff_mean = jnp.array(0.0, dtype=target_values.dtype)

        return target_values, target_advs, retrace_coeff_mean

    def compute_extras(key: Key, train_state: REPPOTrainState, batch: Transition):
        key, act1_key, act2_key, act3_key = jax.random.split(key, 4)

        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(
            train_state.critic.graphdef,
            train_state.critic.params,
        )
        target_critic_model = nnx.merge(
            train_state.target_critic.graphdef,
            train_state.target_critic.params,
        )
        actor_model.eval()
        critic_model.eval()
        target_critic_model.eval()

        next_pi = actor_model(batch.next_obs)
        td_next_action, log_probs = next_pi.sample_and_log_prob(seed=act1_key)

        critic_output = critic_model(batch.next_obs, td_next_action)
        target_critic_output = target_critic_model(batch.next_obs, td_next_action)

        if getattr(hparams, "use_one_step_td", True):
            value = target_critic_output["value"]
        else:
            value = critic_output["value"]
        next_emb = critic_output["embed"]

        # compute average policy value
        if hparams.scale_samples_with_action_d:
            num_samples = 8 * d
        else:
            num_samples = 8      

        # for retrace
        # Q(st, at)
        critic_action = batch.action.clip(-0.999, 0.999) if isinstance(action_space, Box) else batch.action
        action_value = critic_model(batch.obs, critic_action)["value"]

        # E[Q(st+1, .)]
        next_actions = next_pi.sample(
            seed=act3_key, sample_shape=(num_samples,)
        )
        next_actions = jnp.clip(next_actions, -0.999, 0.999)
        next_obs = jnp.repeat(batch.next_obs[None, ...], next_actions.shape[0], axis=0)
        next_policy_value = critic_model(next_obs, next_actions)["value"].mean(0)

        soft_reward = (
            batch.reward - hparams.gamma * log_probs * actor_model.temperature()
        )

        # E[Q(st, .)]
        pi = actor_model(batch.obs)
        current_log_probs = pi.log_prob(batch.action.clip(-0.999, 0.999))
        actions = pi.sample(
            seed=act2_key, sample_shape=(num_samples,)
        )  # WARNING: magic number
        actions = jnp.clip(actions, -0.999, 0.999)
        obs = jnp.repeat(batch.obs[None, ...], actions.shape[0], axis=0)
        policy_value = critic_model(obs, actions)["value"].mean(0)

        extras = {
            "soft_reward": soft_reward * cfg.env.get("reward_scaling", 1.0),
            "value": value,
            "action_value": action_value,
            "policy_value": policy_value,
            "next_policy_value": next_policy_value,
            "next_emb": next_emb,
            "next_actions": td_next_action,
            "log_prob": current_log_probs,
            "current_log_prob": current_log_probs,
        }
        return extras

    def learner_fn(
        key: Key, train_state: REPPOTrainState, batch: Transition
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        initial_obs = train_state.params["sr_dice_initial_obs"]
        if hparams.normalize_env:
            initial_obs = normalizer.normalize(
                train_state.normalization_state, initial_obs
            )
            batch = batch.replace(
                obs=normalizer.normalize(train_state.normalization_state, batch.obs),
                next_obs=normalizer.normalize(
                    train_state.normalization_state, batch.next_obs
                ),
            )

        # compute n-step lambda estimates
        key, act_key = jax.random.split(key)
        extras = compute_extras(key=act_key, train_state=train_state, batch=batch)
        batch.extras.update(extras)

        # compute log probs and ESS
        current_log_prob = batch.extras["current_log_prob"]
        behavior_log_prob = batch.extras["behavior_log_prob"]
        log_importance_ratio = current_log_prob - behavior_log_prob
        importance_ratio = jnp.exp(jnp.clip(log_importance_ratio, -20.0, 20.0))
        mean_importance_ratio = importance_ratio.mean()
        importance_ratio_ess = jnp.square(importance_ratio.sum()) / (
            jnp.square(importance_ratio).sum() + 1e-8
        )
        reward_mean = batch.reward.mean()
        policy_log_prob_mean = current_log_prob.mean()

        (
            batch.extras["target_values"],
            batch.extras["target_advs"],
            retrace_coeff_mean,
        ) = nstep_lambda(batch=batch)

        # Per-env-slot TD error for PER priority updates
        # Shape: [num_envs].
        per_env_td_error = jnp.abs(batch.extras["action_value"] - batch.extras["target_values"]).mean(axis=0)

        # Flatten the sampled replay batch; its size is independent of collection size.
        batch_size = batch.obs.shape[0] * batch.obs.shape[1]
        batch = jax.tree.map(
            lambda x: x.reshape((batch_size, *x.shape[2:])),
            batch,
        )
        # J(π_before): already computed in compute_extras using the old actor
        policy_value_before_vec = batch.extras["policy_value"].reshape(-1)
        # Update the model for a number of epochs
        key, train_key, pv_after_key = jax.random.split(key, 3)
        train_state, update_metrics = jax.lax.scan(
            f=lambda train_state, key: run_epoch(key, train_state, batch, initial_obs),
            init=train_state,
            xs=jax.random.split(train_key, hparams.num_epochs),
        )
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)
        
        # J(π_after): updated actor's expected value on the same states
        _n_pv = 8 * d if hparams.scale_samples_with_action_d else 8
        actor_after = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_after = nnx.merge(
            train_state.critic.graphdef,
            train_state.critic.params,
        )
        actor_after.eval()
        critic_after.eval()
        action_after = actor_after(batch.obs).sample(seed=pv_after_key, sample_shape=(_n_pv,))
        action_after = jnp.clip(action_after, -0.999, 0.999)
        obs_tiled = jnp.repeat(batch.obs[None, ...], action_after.shape[0], axis=0)
        policy_value_after_vec = critic_after(obs_tiled, action_after)["value"].mean(0).reshape(-1)
        policy_improvement = jnp.mean(
            policy_value_after_vec - policy_value_before_vec
        )

        base_metrics = {
            "sys/grad_updates_per_learner_call": hparams.num_epochs * hparams.num_mini_batches,
        }
        update_metrics = {**update_metrics, **base_metrics}
        return train_state, update_metrics, per_env_td_error

    return jax.jit(learner_fn)