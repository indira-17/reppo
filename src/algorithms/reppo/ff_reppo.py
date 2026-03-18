import logging
import optax
import math
from typing import Callable
import operator
import os

import numpy as np
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
import torch

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
    
    logging.debug(f"Loading BC weights from {bc_checkpoint_path}")
    
    try:
        with open(bc_checkpoint_path, 'rb') as f:
            saved_state = pickle.load(f)
        
        nnx.update(jax_actor, saved_state)
        logging.debug(f"Successfully loaded JAX BC weights from {bc_checkpoint_path}")
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

    def __call__(self, key: jax.Array, x: jax.Array, **kwargs) -> distrax.Distribution:
        if self.normalizer is not None:
            x = self.normalizer.normalize(self.normalization_state, x)
        if self._eval_mode:
            action = self.base.det_action(x)
            info = {}
        else:
            pi = self.base(x, **kwargs)
            action, log_prob = pi.sample_and_log_prob(seed=key)
            info = {"log_prob": log_prob}
        if isinstance(self.action_space, Box):
            action = action.clip(-0.999, 0.999)
        return action, info


def make_policy_fn(
    cfg: DictConfig, observation_space: Space, action_space: Space
) -> Callable[[REPPOTrainState, bool], Policy]:
    cfg = cfg.algorithm

    def policy_fn(train_state: REPPOTrainState, eval: bool) -> Policy:
        normalizer = Normalizer() if cfg.normalize_env else None
        actor_state = train_state.actor
        if not eval and train_state.rollout_actor is not None:
            actor_state = train_state.rollout_actor
        actor_model = nnx.merge(actor_state.graphdef, actor_state.params)

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
    print(action_space.shape)

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

        # Initialize the actor with BC weights if available (BC initialization only)
        bc_checkpoint_path = getattr(hparams, "bc_checkpoint_path", None)
        if hparams.bc_indicator and bc_checkpoint_path and os.path.exists(bc_checkpoint_path):
            logging.debug(f"Loading actor weights from {bc_checkpoint_path}")
            actor = load_bc_weights_to_actor(bc_checkpoint_path, actor)
        logging.debug("JAX Actor structure successfully created")
        # Only one actor and one rollout_actor (reference policy)
        return REPPOTrainState.create(
            graphdef=nnx.graphdef(actor),
            params=nnx.state(actor),
            tx=tx,
            actor=nnx.TrainState.create(
                graphdef=nnx.graphdef(actor), params=nnx.state(actor), tx=tx
            ),
            critic=nnx.TrainState.create(
                graphdef=nnx.graphdef(critic), params=nnx.state(critic), tx=tx
            ),
            actor_target=nnx.TrainState.create(
                graphdef=nnx.graphdef(actor), params=nnx.state(actor), tx=tx
            ),
            # rollout_actor is a simple dict, not a TrainState
            rollout_actor={
                'graphdef': nnx.graphdef(actor),
                'params': nnx.state(actor),
            },
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

        # compute l2 error for logging
        critic_loss = optax.squared_error(
            value,
            target_values,
        )
        critic_loss = jnp.mean(critic_loss)
        valid = minibatch.extras.get("valid_mask", jnp.ones_like(minibatch.reward))
        mask_truncated = hparams.mask_truncated
        mask = (1.0 - minibatch.truncated) if mask_truncated else 1.0
        mask = mask * valid  # mask out padded timesteps from offline sampling
        denom = jnp.maximum(jnp.sum(valid), 1.0)
        loss = jnp.sum(
            mask
            * (critic_update_loss + hparams.aux_loss_mult * aux_loss)
        ) / denom
        return loss, dict(
            value_loss=critic_loss,
            critic_update_loss=critic_update_loss,
            loss=loss,
            aux_loss=aux_loss,
            rew_aux_loss=aux_rew_loss,
            q=value.mean(),
            abs_batch_action=jnp.abs(minibatch.action).mean(),
            reward_mean=minibatch.reward.mean(),
            target_values=target_values.mean(),
        )

    def actor_loss(
        params: nnx.Param, train_state: REPPOTrainState, minibatch: Transition
    ):
        valid = minibatch.extras.get("valid_mask", jnp.ones(minibatch.reward.shape, dtype=jnp.float32))

        def masked_mean(x):
            x = jnp.asarray(x)
            denom = jnp.maximum(jnp.sum(valid), 1.0)
            return jnp.sum(x * valid) / denom

        critic_target_model = nnx.merge(
            train_state.critic.graphdef,
            train_state.critic.params,
        )
        actor_model = nnx.merge(train_state.actor.graphdef, params)
        # Use rollout_actor as the reference policy for KL
        # set up models for training with batch norm
        actor_model.train()
        critic_target_model.eval()
        rollout_actor = nnx.merge(train_state.rollout_actor['graphdef'], train_state.rollout_actor['params'])
        rollout_actor.eval()
        pi = actor_model(minibatch.obs)
        old_pi = rollout_actor(minibatch.obs)
        # policy KL with independent linear schedulers:
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
            loss = masked_mean(
                actor_loss + kl * jax.lax.stop_gradient(lagrangian) * hparams.reduce_kl
            )
        elif hparams.actor_kl_clip_mode == "clipped":
            loss = masked_mean(
                jnp.where(
                    kl < hparams.kl_bound,
                    actor_loss,
                    kl * jax.lax.stop_gradient(lagrangian) * hparams.reduce_kl,
                )
            )
        elif hparams.actor_kl_clip_mode == "value":
            loss = masked_mean(actor_loss)
        else:
            raise ValueError(f"Unknown actor loss mode: {hparams.actor_kl_clip_mode}")

        # SAC target entropy loss

        target_entropy = action_size_target + entropy
        target_entropy_loss = actor_model.temperature() * jax.lax.stop_gradient(
            target_entropy
        )

        # Lagrangian constraint (follows temperature update)
        lagrangian_loss = -lagrangian * jax.lax.stop_gradient(kl - hparams.kl_bound)

        # total loss
        if hparams.update_entropy_lagrangian:
            loss += masked_mean(target_entropy_loss)
        if hparams.update_kl_lagrangian:
            loss += masked_mean(lagrangian_loss)

        # for logging
        real_action_log_prob = old_pi.log_prob(
            minibatch.action.clip(-0.999, 0.999)
        ).mean()

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

    # rollout_actor_tx will be set in make_init_fn and carried in the state, so we can reference it from train_state.rollout_actor.tx
    def update(train_state: REPPOTrainState, batch: Transition):
        critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
        output, grads = critic_grad_fn(train_state.critic.params, train_state, batch)
        critic_train_state = train_state.critic.apply_gradients(grads)
        train_state = train_state.replace(
            critic=critic_train_state,
        )
        critic_metrics = output[1]

        if hparams.bc_indicator:
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
                _, actor_metrics = actor_loss(train_state.actor.params, train_state, batch)
                return actor_train_state, grad_norm, actor_metrics

            actor_update_delay = int(getattr(hparams, "bc_actor_update_delay", 0))
            actor_train_state, grad_norm, actor_metrics = jax.lax.cond(
                train_state.iteration >= actor_update_delay,
                update_actor,
                hold_update_actor,
                operand=None,
            )
        else:
            actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
            output, grads = actor_grad_fn(train_state.actor.params, train_state, batch)
            grad_norm = jax.tree.map(lambda x: jnp.linalg.norm(x), grads)
            grad_norm = jax.tree.reduce(operator.add, grad_norm)
            grads = jax.tree.map(
                lambda x: jnp.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0), grads
            )
            actor_train_state = train_state.actor.apply_gradients(grads)
            actor_metrics = output[1]

        # Only update actor; rollout_actor is frozen at the start of each outer
        # iteration to serve as the KL reference policy.
        train_state = train_state.replace(
            actor=actor_train_state,
        )
        return train_state, {
            **critic_metrics,
            **actor_metrics,
            "grad_norm": grad_norm,
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
        # Compute max metrics across mini-batches
        # metrics_max = jax.tree.map(lambda x: x.max(), metrics)
        # metrics_min = jax.tree.map(lambda x: x.min(), metrics)
        return (
            train_state,
            metrics_mean,
        )  # {**metrics_mean, **{k + "_max": v for k, v in metrics_max.items()}, **{k + "_min": v for k, v in metrics_min.items()}}

    def nstep_lambda(batch: Transition):
        def loop(carry: tuple[jax.Array, ...], transition: Transition):
            retrace_target, gae, next_value, next_c, next_q = carry
            done = transition.done
            truncated = transition.truncated
            reward = transition.extras["soft_reward"]
            expected_next_q = transition.extras["next_value"]
            q_value = transition.extras["q_value"]
            policy_value = transition.extras["policy_value"]
            log_prob_behavior = transition.extras["behavior_log_prob"]
            log_prob_current = transition.extras["log_prob_current"]

            log_ratio = log_prob_current - log_prob_behavior
            ratio = jnp.exp(jnp.clip(log_ratio, -5.0, 5.0))
            c_t = hparams.lmbda * jnp.minimum(1.0, ratio)

            td_error = reward + hparams.gamma * (1.0 - done) * expected_next_q - q_value
            retrace_target = q_value + td_error + hparams.gamma * (1.0 - done) * next_c * (retrace_target - next_q)
            retrace_target = jnp.where(truncated, q_value + td_error, retrace_target)

            delta = reward + hparams.gamma * (1.0 - done) * next_value - policy_value
            gae = delta + hparams.gamma * (1.0 - done) * hparams.lmbda * gae
            gae = jnp.where(truncated, reward + hparams.gamma * (1.0 - done) * next_value - expected_next_q, gae)

            return (
                retrace_target,
                gae,
                expected_next_q,
                c_t,
                q_value,
            ), (retrace_target, gae)

        _, (target_values, target_advs) = jax.lax.scan(
            f=loop,
            init=(
                batch.extras["next_value"][-1],
                batch.extras["policy_value"][-1],
                batch.extras["next_value"][-1],
                jnp.ones_like(batch.extras["next_value"][-1]),
                batch.extras["q_value"][-1],
            ),
            xs=batch,
            reverse=True,
        )
        return target_values, target_advs

    def compute_extras(key: Key, train_state: REPPOTrainState, batch: Transition):
        key, act1_key, act2_key, act3_key = jax.random.split(key, 4)

        actor_model = nnx.merge(train_state.actor.graphdef, train_state.actor.params)
        critic_model = nnx.merge(train_state.critic.graphdef, train_state.critic.params)
        actor_model.eval()
        critic_model.eval()

        actions, log_probs = actor_model(batch.next_obs).sample_and_log_prob(
            seed=act1_key
        )
        actions = jnp.clip(actions, -0.999, 0.999)
        critic_output = critic_model(batch.next_obs, actions)
        next_emb = critic_output["embed"]

        soft_reward = (
            batch.reward - hparams.gamma * log_probs * actor_model.temperature()
        )

        if hparams.scale_samples_with_action_d:
            num_samples = 8 * d
        else:
            num_samples = 8
        next_pi = actor_model(batch.next_obs)
        next_actions = next_pi.sample(seed=act2_key, sample_shape=(num_samples,))
        next_actions = jnp.clip(next_actions, -0.999, 0.999)
        next_obs = jnp.repeat(batch.next_obs[None, ...], next_actions.shape[0], axis=0)
        next_q_values = critic_model(next_obs, next_actions)["value"]
        next_value = next_q_values.mean(0)

        # compute average policy value for the actor baseline
        pi = actor_model(batch.obs)
        actions = pi.sample(
            seed=act3_key, sample_shape=(num_samples,)
        )
        actions = jnp.clip(actions, -0.999, 0.999)
        obs = jnp.repeat(batch.obs[None, ...], actions.shape[0], axis=0)
        policy_value = critic_model(obs, actions)["value"].mean(0)

        # Compute current policy log prob of actual batch actions for retrace importance ratio
        pi_current = actor_model(batch.obs)
        batch_action = jnp.clip(batch.action, -0.999, 0.999)
        log_prob_current = pi_current.log_prob(batch_action)
        q_value = critic_model(batch.obs, batch_action)["value"]

        # For offline data, use current policy log probs so retrace importance
        # ratios stay near 1 instead of exploding as the policy drifts.
        is_offline = batch.extras.get("is_offline", jnp.zeros_like(batch.reward))
        behavior_log_prob = jnp.where(
            is_offline > 0.5,
            log_prob_current,
            batch.extras["behavior_log_prob"],
        )

        extras = {
            "soft_reward": soft_reward * cfg.env.get("reward_scaling", 1.0),
            "next_value": next_value,
            "policy_value": policy_value,
            "next_emb": next_emb,
            "log_prob": log_probs,
            "log_prob_current": log_prob_current,
            "behavior_log_prob": behavior_log_prob,
            "q_value": q_value,
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

        batch.extras["target_values"], batch.extras["target_advs"] = nstep_lambda(
            batch=batch
        )

        # Reshape data to (num_steps * num_envs, ...)

        batch = jax.tree.map(
            lambda x: x.reshape((hparams.num_steps * hparams.num_envs, *x.shape[2:])),
            batch,
        )
        train_state = train_state.replace(
            actor_target=train_state.actor_target.replace(
                params=train_state.actor.params
            ),
        )
        # Update the model for a number of epochs
        key, train_key = jax.random.split(key)
        train_state, update_metrics = jax.lax.scan(
            f=lambda train_state, key: run_epoch(key, train_state, batch),
            init=train_state,
            xs=jax.random.split(train_key, hparams.num_epochs),
        )
        # Get metrics from the last epoch
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)
        return train_state, update_metrics

    return jax.jit(learner_fn)