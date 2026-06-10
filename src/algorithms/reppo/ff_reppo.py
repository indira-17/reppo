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

    def __call__(self, key: jax.Array, x: jax.Array, **kwargs) -> distrax.Distribution:
        if self.normalizer is not None:
            x = self.normalizer.normalize(self.normalization_state, x)
        if self._eval_mode:
            action = self.base.det_action(x)
        else:
            pi = self.base(x, **kwargs)
            action = pi.sample(seed=key)
        if isinstance(self.action_space, Box):
            action = action.clip(-0.999, 0.999)
        return action, {}


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

        return REPPOTrainState.create(
            graphdef=nnx.graphdef(actor),
            params=nnx.state(actor),
            tx=optax.set_to_zero(),
            actor=nnx.TrainState.create(
                graphdef=nnx.graphdef(actor), params=nnx.state(actor), tx=actor_tx
            ),
            critic=nnx.TrainState.create(
                graphdef=nnx.graphdef(critic), params=nnx.state(critic), tx=tx
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
    per_beta = getattr(hparams, 'per_beta', 0.4)
    discrete_actions = isinstance(action_space, Discrete)
    d = action_space.shape[-1] if not discrete_actions else action_space.n

    def critic_loss_fn(
        params: nnx.Param, train_state: REPPOTrainState, minibatch: Transition
    ):
        critic_model = nnx.merge(train_state.critic.graphdef, params)
        critic_model.train()
        critic_output = critic_model(minibatch.obs, minibatch.action)

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

        # Aux loss
        pred = critic_output["pred_features"]
        pred_rew = critic_output["pred_rew"]
        value = critic_output["value"]
        aux_loss = optax.squared_error(pred, minibatch.extras["next_emb"])
        aux_rew_loss = optax.squared_error(pred_rew, minibatch.reward.reshape(-1, 1))
        aux_loss = jnp.mean(
            (1 - minibatch.done.reshape(-1, 1))
            * jnp.concatenate([aux_loss, aux_rew_loss], axis=-1),
            axis=-1,
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
        unmasked_critic_total_loss = jnp.mean(
            critic_update_loss + hparams.aux_loss_mult * aux_loss
        )
        loss = jnp.mean(
            per_scale * mask
            * (critic_update_loss + hparams.aux_loss_mult * aux_loss)
        )
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
            rew_aux_loss=aux_rew_loss,
            abs_batch_action=jnp.abs(minibatch.action).mean(),
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
            temp=actor_model.temperature(),
            abs_pred_action=jnp.abs(pred_action).mean()
            if not discrete_actions
            else 0.0,
            lagrangian=lagrangian,
            lagrangian_loss=lagrangian_loss,
            entropy_loss=target_entropy_loss,
            # --- actor diagnostics ---
            **{
                "actor_diag/kl": kl.mean(),
                "actor_diag/entropy": jnp.mean(entropy),
                "actor_diag/actor_loss": jnp.mean(actor_loss),
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

    def update(train_state: REPPOTrainState, batch: Transition):
        # Update critic always
        if cfg.algorithm.bc_indicator:
            def update_critic(_):
                critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                output, grads = critic_grad_fn(train_state.critic.params, train_state, batch)
                critic_train_state = train_state.critic.apply_gradients(grads)
                critic_metrics = output[1]
                return critic_train_state, critic_metrics
            
            # Always update the critic
            critic_train_state, critic_metrics = update_critic(None)
            train_state = train_state.replace(critic=critic_train_state)
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
            critic_train_state = train_state.critic.apply_gradients(grads)
            train_state = train_state.replace(
                critic=critic_train_state,
            )
            critic_metrics = output[1]

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
            "actor_diag/grad_norm": grad_norm
        }

    def run_epoch(
        key: jax.Array, train_state: REPPOTrainState, batch: Transition
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        # Shuffle data and split into mini-batches
        key, shuffle_key, act_key, kl_key = jax.random.split(key, 4)
        mini_batch_size = (
            math.floor(hparams.num_steps * hparams.num_envs) // hparams.num_mini_batches
        )
        indices = jax.random.permutation(
            shuffle_key, hparams.num_steps * hparams.num_envs
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
        # Gradient variance: Var(||g_θ||) across mini-batches
        metrics_mean["actor_diag/grad_norm_var"] = jnp.var(metrics["actor_diag/grad_norm"])
        # Compute max metrics across mini-batches
        # metrics_max = jax.tree.map(lambda x: x.max(), metrics)
        # metrics_min = jax.tree.map(lambda x: x.min(), metrics)
        return (
            train_state,
            metrics_mean,
        )  # {**metrics_mean, **{k + "_max": v for k, v in metrics_max.items()}, **{k + "_min": v for k, v in metrics_min.items()}}

    def nstep_lambda(batch: Transition):
        if not getattr(hparams, "use_retrace", True):
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
            # Retrace(lambda) for BC policy
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
                # G_t = r_tilde[t] + gamma * (1 - d[t]) * (V[t+1] + c_{t+1} * (G_{t+1} - Q(x[t+1], a[t+1])))
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
                    batch.extras["next_policy_value"][-1],
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
        actor_model.eval()
        critic_model.eval()

        actions, log_probs = actor_model(batch.next_obs).sample_and_log_prob(
            seed=act1_key
        )
        critic_output = critic_model(batch.next_obs, actions)
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
        next_pi = actor_model(batch.next_obs)
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
            "log_prob": current_log_probs,
            "current_log_prob": current_log_probs,
        }
        return extras

    def learner_fn(
        key: Key, train_state: REPPOTrainState, batch: Transition
    ) -> tuple[REPPOTrainState, dict[str, jax.Array]]:
        if hparams.normalize_env:
            new_norm_state = normalizer.update(
                train_state.normalization_state, batch.obs
            )
            batch = batch.replace(
                obs=normalizer.normalize(train_state.normalization_state, batch.obs),
                next_obs=normalizer.normalize(
                    train_state.normalization_state, batch.next_obs
                ),
            )
            train_state = train_state.replace(normalization_state=new_norm_state)

        # compute n-step lambda estimates
        key, act_key = jax.random.split(key)
        extras = compute_extras(key=act_key, train_state=train_state, batch=batch)
        batch.extras.update(extras)

        # compute log probs; split by source when offline/online data is mixed
        current_log_prob = batch.extras["current_log_prob"]
        reward_mean = batch.reward.mean()
        policy_log_prob_mean = current_log_prob.mean()
        source_is_offline = batch.extras.get("source_is_offline")
        if source_is_offline is not None:
            offline_count = source_is_offline.sum()
            online_mask = 1.0 - source_is_offline
            online_count = online_mask.sum()
            reward_mean_offline = jnp.where(
                offline_count > 0,
                (batch.reward * source_is_offline).sum() / offline_count,
                0.0,
            )
            reward_mean_online = jnp.where(
                online_count > 0,
                (batch.reward * online_mask).sum() / online_count,
                0.0,
            )
            policy_log_prob_offline = jnp.where(
                offline_count > 0,
                (current_log_prob * source_is_offline).sum() / offline_count,
                0.0,
            )
            policy_log_prob_online = jnp.where(
                online_count > 0,
                (current_log_prob * online_mask).sum() / online_count,
                0.0,
            )
        else:
            reward_mean_offline = jnp.array(0.0, dtype=reward_mean.dtype)
            reward_mean_online = reward_mean
            policy_log_prob_offline = jnp.array(0.0, dtype=policy_log_prob_mean.dtype)
            policy_log_prob_online = policy_log_prob_mean

        (
            batch.extras["target_values"],
            batch.extras["target_advs"],
            retrace_coeff_mean,
        ) = nstep_lambda(batch=batch)

        # Per-env-slot TD error for PER priority updates
        # Shape: [num_envs]. Offline segments occupy slots 0..num_offline-1.
        per_env_td_error = jnp.abs(batch.extras["action_value"] - batch.extras["target_values"]).mean(axis=0)

        # Reshape data to (num_steps * num_envs, ...)
        batch = jax.tree.map(
            lambda x: x.reshape((hparams.num_steps * hparams.num_envs, *x.shape[2:])),
            batch,
        )
        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(
                params=train_state.actor.params
            )
        )
        # J(π_before): already computed in compute_extras using the old actor
        policy_value_before_vec = batch.extras["policy_value"].reshape(-1)
        # Update the model for a number of epochs
        key, train_key, pv_after_key = jax.random.split(key, 3)
        train_state, update_metrics = jax.lax.scan(
            f=lambda train_state, key: run_epoch(key, train_state, batch),
            init=train_state,
            xs=jax.random.split(train_key, hparams.num_epochs),
        )
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)
        
        # J(π_after): updated actor's expected value on the same states
        _n_pv = 8 * d if hparams.scale_samples_with_action_d else 8
        actor_after = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_after = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        actor_after.eval()
        critic_after.eval()
        action_after = actor_after(batch.obs).sample(seed=pv_after_key, sample_shape=(_n_pv,))
        action_after = jnp.clip(action_after, -0.999, 0.999)
        obs_tiled = jnp.repeat(batch.obs[None, ...], action_after.shape[0], axis=0)
        policy_value_after_vec = critic_after(obs_tiled, action_after)["value"].mean(0).reshape(-1)
        policy_improvement_online, policy_improvement_offline = _split_online_offline_mean(
            policy_value_after_vec - policy_value_before_vec,
            batch.extras.get("source_is_offline"),
        )

        base_metrics = {
            "reward_mean": reward_mean,
            "actor_diag/policy_log_prob_mean": policy_log_prob_mean,
            # retrace coefficient: measure of off-policyness (1.0 = on-policy, < 1.0 = off-policy)
            "critic_diag/retrace_coeff_mean": retrace_coeff_mean,
            "actor_diag/retrace_coeff_mean": retrace_coeff_mean,
            "actor_diag/policy_improvement": policy_improvement_online,
            "sys/grad_updates": (train_state.time_steps // (hparams.num_steps * hparams.num_envs)) * hparams.num_epochs * hparams.num_mini_batches,
        }
        # Per-source split only for the mixed-source (expert) path.
        if source_is_offline is not None:
            base_metrics.update({
                "reward_mean_offline": reward_mean_offline,
                "reward_mean_online": reward_mean_online,
                "actor_diag/policy_log_prob_offline_replay": policy_log_prob_offline,
                "actor_diag/policy_log_prob_online_replay": policy_log_prob_online,
                "actor_diag/policy_improvement_offline": policy_improvement_offline,
            })
        update_metrics = {**update_metrics, **base_metrics}
        return train_state, update_metrics, per_env_td_error

    return jax.jit(learner_fn)