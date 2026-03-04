import os
import math
import pickle
import numpy as np
import hydra
import jax
import jax.numpy as jnp
import optax
import distrax
import matplotlib.pyplot as plt
import gymnasium as gym
import wandb
import torch
from datetime import datetime
from flax import nnx
from omegaconf import OmegaConf

from src.networks.common import MLP
from src.networks.policy_heads import TanhGaussianPolicyHead
from src.algorithms.reppo.networks import Actor
from src.maniskill_utils.maniskill_dataloader_shabnam import load_demos_for_training


def create_actor(n_obs, n_act, cfg, rngs):
    """Create a JAX Actor matching the REPPO architecture."""
    hparams = cfg.algorithm
    feature_encoder = MLP(
        in_features=n_obs,
        out_features=n_act * 2,
        hidden_dim=hparams.actor_hidden_dim,
        hidden_activation=nnx.swish,
        output_activation=None,
        use_norm=hparams.use_actor_norm,
        use_output_norm=False,
        layers=hparams.num_actor_layers,
        hidden_skip=hparams.use_actor_skip,
        output_skip=hparams.use_actor_skip,
        rngs=rngs,
    )
    actor = Actor(
        feature_encoder=feature_encoder,
        policy_head=TanhGaussianPolicyHead(
            min_std=hparams.actor_min_std,
            fixed_std=hparams.fixed_actor_std,
        ),
        kl_start=hparams.kl_start,
        ent_start=hparams.ent_start,
    )
    return actor


def compute_loss(actor, obs, expert_action, bc_mse_loss_weight):
    """Compute combined NLL + MSE loss for BC training."""
    dist = actor(obs)

    # NLL loss
    log_prob = dist.log_prob(expert_action)  # [B]
    nll_loss = -log_prob.mean()

    # MSE loss: need tanh(mean) vs expert_action
    tanh_mean = actor.det_action(obs)  # [B, action_dim]
    mse_loss = jnp.mean((tanh_mean - expert_action) ** 2)

    # For diagnostics: extract mean and log_std from features
    features = actor.feature_encoder(obs)
    mean, log_std = jnp.split(features, 2, axis=-1)

    total_loss = nll_loss + bc_mse_loss_weight * mse_loss

    metrics = {
        "nll_loss": nll_loss,
        "mse_loss": mse_loss,
        "mean": jnp.mean(mean),
        "tanh_mean": jnp.mean(tanh_mean),
        "log_std": jnp.mean(log_std),
    }
    return total_loss, metrics


@nnx.jit
def train_step(actor, opt_state, obs, expert_action, bc_mse_loss_weight):
    """Single JIT-compiled training step."""
    (loss, metrics), grads = nnx.value_and_grad(compute_loss, has_aux=True)(
        actor, obs, expert_action, bc_mse_loss_weight
    )
    opt_state.update(actor, grads)
    return loss, metrics


def train_one_epoch(cfg, train_loader, actor, opt_state, data_low, data_high):
    """Train one epoch and return average metrics."""
    actor_loss_sum = 0.0
    nll_loss_sum = 0.0
    mse_loss_sum = 0.0
    mean_sum = 0.0
    tanh_mean_sum = 0.0
    log_std_sum = 0.0
    num_batches = 0

    bc_mse_loss_weight = cfg.algorithm.bc_mse_loss_weight

    for data in train_loader:
        obs = data['observations'].numpy()
        expert_action = data['actions'].numpy()

        # Normalize expert actions to [-1, 1] using bounds with safety margin
        expert_action = 2.0 * (expert_action - data_low) / (data_high - data_low) - 1.0

        # Convert to jax arrays
        obs = jnp.array(obs)
        expert_action = jnp.array(expert_action)

        loss, metrics = train_step(actor, opt_state, obs, expert_action, bc_mse_loss_weight)

        actor_loss_sum += float(loss)
        nll_loss_sum += float(metrics["nll_loss"])
        mse_loss_sum += float(metrics["mse_loss"])
        mean_sum += float(metrics["mean"])
        tanh_mean_sum += float(metrics["tanh_mean"])
        log_std_sum += float(metrics["log_std"])
        num_batches += 1

    return (
        actor_loss_sum / num_batches,
        nll_loss_sum / num_batches,
        mse_loss_sum / num_batches,
        mean_sum / num_batches,
        tanh_mean_sum / num_batches,
        log_std_sum / num_batches,
    )


def evaluate(cfg, val_loader, actor, data_low, data_high):
    """Evaluate on validation set and return average metrics."""
    val_loss_sum = 0.0
    val_nll_loss = 0.0
    val_mse_loss = 0.0
    val_mean = 0.0
    val_tanh_mean = 0.0
    val_log_std = 0.0
    val_num_batches = 0

    bc_mse_loss_weight = cfg.algorithm.bc_mse_loss_weight

    for vdata in val_loader:
        obs = jnp.array(vdata['observations'].numpy())
        expert_action = vdata['actions'].numpy()

        # Normalize expert actions to [-1, 1]
        expert_action = 2.0 * (expert_action - data_low) / (data_high - data_low) - 1.0
        expert_action = jnp.array(expert_action)

        # Forward pass (no gradient needed — JAX is functional, just call)
        dist = actor(obs)
        log_prob = dist.log_prob(expert_action)

        nll_loss = -log_prob.mean()
        tanh_mean = actor.det_action(obs)
        mse_loss = jnp.mean((tanh_mean - expert_action) ** 2)

        features = actor.feature_encoder(obs)
        mean, log_std = jnp.split(features, 2, axis=-1)

        val_loss = nll_loss + bc_mse_loss_weight * mse_loss

        val_loss_sum += float(val_loss)
        val_nll_loss += float(nll_loss)
        val_mse_loss += float(mse_loss)
        val_mean += float(jnp.mean(mean))
        val_tanh_mean += float(jnp.mean(tanh_mean))
        val_log_std += float(jnp.mean(log_std))
        val_num_batches += 1

    return (
        val_loss_sum / val_num_batches,
        val_nll_loss / val_num_batches,
        val_mse_loss / val_num_batches,
        val_mean / val_num_batches,
        val_tanh_mean / val_num_batches,
        val_log_std / val_num_batches,
    )


def save_actor(actor, path):
    """Save JAX actor state using nnx.state + pickle."""
    _, state = nnx.split(actor)
    with open(path, 'wb') as f:
        pickle.dump(jax.device_get(state), f)
    print(f"  ✓ Saved JAX actor to {path}")


def load_actor(actor, path):
    """Load JAX actor state from pickle file."""
    with open(path, 'rb') as f:
        state = pickle.load(f)
    nnx.update(actor, state)
    return actor


@hydra.main(version_base=None, config_path="../config/default", config_name="reppo_maniskill")
def main(cfg: OmegaConf):
    """Main training function."""
    OmegaConf.set_struct(cfg, False)

    # Initialize wandb
    wandb.init(
        config=dict(cfg),
        entity=cfg.logging.entity,
        project="reppo",
        name=f"bc_pretrain_jax_no_clamp_linear_scale_no_envstates_{cfg.env.name}",
        mode=cfg.logging.mode,
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    EPOCHS = 100

    # Load demo parameters from config
    env_name = cfg.env.name
    batch_size = cfg.env.demo.batch_size
    demo_path = cfg.env.demo.demo_path
    max_episodes = cfg.env.demo.max_episodes
    filter_success = cfg.env.demo.filter_success

    # Load demonstrations (still uses torch DataLoader internally)
    device = 'cpu'  # load data on CPU, convert to JAX later
    train_loader, val_loader, n_obs, n_act, _, _ = load_demos_for_training(
        env_id=env_name,
        device=device,
        bsize=batch_size,
        demo_path=demo_path,
        max_episodes=max_episodes,
        filter_success=filter_success,
    )

    # Create JAX actor
    rngs = nnx.Rngs(0)
    actor = create_actor(n_obs, n_act, cfg, rngs)

    # Get action bounds from environment
    temp_env = gym.make(env_name, obs_mode="state_dict", control_mode=cfg.env.get('control_mode'))
    low = temp_env.action_space.low
    high = temp_env.action_space.high
    temp_env.close()
    print(f"True environment bounds: low={low}, high={high}")

    # Create optax optimizer
    optimizer = optax.adamw(learning_rate=float(cfg.algorithm.optimizer.learning_rate))
    opt_state = nnx.Optimizer(actor, optimizer, wrt=nnx.Param)

    # Compute dataset action stats for normalization bounds
    all_actions = np.concatenate(
        [batch["actions"].numpy() for batch in train_loader] +
        [batch['actions'].numpy() for batch in val_loader],
        axis=0,
    )

    data_low = all_actions.min(axis=0)
    data_high = all_actions.max(axis=0)

    # Add 10% safety margin to bounds
    margin = 0.1 * (data_high - data_low)
    low_with_margin = data_low - margin
    high_with_margin = data_high + margin

    # Ensure bounds are at least [-1, 1] in each dimension
    low = np.minimum(low_with_margin, -1.0)
    high = np.maximum(high_with_margin, 1.0)

    # Normalize actions for stats check
    normalized_actions = 2.0 * (all_actions - low) / (high - low) - 1.0

    print("\n=== DATASET ACTION STATS ===")
    print(f"Expert action min (raw): {all_actions.min():.6f}")
    print(f"Expert action max (raw): {all_actions.max():.6f}")
    print(f"Normalized action min: {normalized_actions.min():.6f}")
    print(f"Normalized action max: {normalized_actions.max():.6f}")
    print(f"Action normalization bounds (low): {low}")
    print(f"Action normalization bounds (high): {high}")
    print("================================\n")

    best_vloss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        print(f'\nEPOCH {epoch + 1}/{EPOCHS}')

        # Training
        avg_actor_loss, avg_nll_loss, avg_mse_loss, avg_mean, avg_tanh_mean, avg_log_std = \
            train_one_epoch(cfg, train_loader, actor, opt_state, low, high)
        train_losses.append(avg_actor_loss)

        # Validation
        avg_val_loss, v_nll_loss, v_mse_loss, v_mean, v_tanh_mean, v_log_std = \
            evaluate(cfg, val_loader, actor, low, high)
        val_losses.append(avg_val_loss)

        print(f'Loss: train={avg_actor_loss:.4f} | val={avg_val_loss:.4f}')

        # Log to wandb
        wandb.log({
            "train/nll_loss": avg_nll_loss,
            "train/mse_loss": avg_mse_loss,
            "train/mean": avg_mean,
            "train/tanh_mean": avg_tanh_mean,
            "train/log_std": avg_log_std,
            "val/nll_loss": v_nll_loss,
            "val/mse_loss": v_mse_loss,
            "val/mean": v_mean,
            "val/tanh_mean": v_tanh_mean,
            "val/log_std": v_log_std,
            "epoch": epoch + 1,
        }, step=epoch + 1)

        # Save best actor based on validation loss
        if avg_val_loss < best_vloss:
            best_vloss = avg_val_loss
            model_dir = f'{env_name}/saved_models/{timestamp}'
            os.makedirs(model_dir, exist_ok=True)
            model_path = f'{model_dir}/bc_model_actor_best.pkl'
            save_actor(actor, model_path)
            print(f'  Best val_loss={avg_val_loss:.4f}')

    print(f"\n{'='*60}")
    print(f"Training Complete! Best validation loss: {best_vloss:.4f}")
    print(f"{'='*60}")

    # Save final loss plots
    plot_dir = f'{env_name}/loss_plots/{timestamp}'
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label=f"Train Loss for {env_name}", color="blue")
    plt.plot(val_losses, label=f"Val Loss for {env_name}", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Actor Loss")
    plt.legend()
    plt.title(f'Losses for {env_name}')
    plot_path = f'{plot_dir}/loss_plot.png'
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved loss plots to {plot_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
