import os
import pickle
import random
import numpy as np
import hydra
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import gymnasium as gym
import mani_skill.envs  # noqa: F401  # Registers ManiSkill envs with Gymnasium.
import wandb
import torch
from datetime import datetime
from flax import nnx
from omegaconf import OmegaConf

from src.networks.common import MLP
from src.networks.policy_heads import TanhGaussianPolicyHead
from src.algorithms.reppo.networks import Actor
from src.maniskill_utils.maniskill_dataloader_shabnam import load_demos_for_training, DemoConfig, ManiSkillDemoLoader

def set_all_seeds(seed: int):
    """Set seeds across python, numpy, torch, and JAX-related code paths."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_pretrain_seeds(cfg):
    """Resolve seed list from config, defaulting to 5 seeds."""
    if "pretrain_seeds" in cfg.algorithm and cfg.algorithm.pretrain_seeds is not None:
        return [int(s) for s in cfg.algorithm.pretrain_seeds]
    return [0]


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


def compute_loss(actor, obs, expert_action, bc_use_entropy, bc_use_entropy_tuning, ent_target_mult, seed):
    """Compute combined NLL + Entropy loss for BC training."""
    dist = actor(obs)

    # Ea~mu[- log pi(a|s)]
    log_prob = dist.log_prob(expert_action)  # [B]
    nll_loss = -log_prob.mean()

    # Ea~pi[- log pi(a|s)]
    _, log_prob_sample = dist.sample_and_log_prob(seed=jax.random.PRNGKey(seed))
    entropy = -log_prob_sample.mean()

    target_entropy_loss = 0.0

    if bc_use_entropy and not bc_use_entropy_tuning:
        # Fixed-alpha entropy: maximize entropy with stop-gradient temperature
        alpha = jax.lax.stop_gradient(actor.temperature())
        total_loss = nll_loss - alpha * entropy

    elif bc_use_entropy and bc_use_entropy_tuning:
        # Learned temperature entropy tuning
        action_size_target = expert_action.shape[-1] * ent_target_mult
        target_entropy = action_size_target + entropy
        target_entropy_loss = actor.temperature() * jax.lax.stop_gradient(target_entropy)
        target_entropy_loss = jnp.squeeze(target_entropy_loss)
        total_loss = nll_loss + target_entropy_loss

    else:
        # No entropy
        total_loss = nll_loss

    total_loss = jnp.squeeze(total_loss)

    # For diagnostics: extract mean and log_std from features
    features = actor.feature_encoder(obs)
    mean, log_std = jnp.split(features, 2, axis=-1)

    metrics = {
        "nll_loss": nll_loss,
        "total_loss": total_loss,
        "mean": jnp.mean(mean),
        "log_std": jnp.mean(log_std),
        "entropy": entropy,
        "entropy_loss": target_entropy_loss,
    }
    return total_loss, metrics


def train_one_epoch(cfg, train_loader, actor, opt_state, data_low, data_high, seed):
    """Train one epoch and return average metrics."""
    actor_loss_sum = 0.0
    nll_loss_sum = 0.0
    entropy_sum = 0.0
    entropy_loss_sum = 0.0
    mean_sum = 0.0
    log_std_sum = 0.0
    num_batches = 0

    bc_use_entropy = bool(cfg.algorithm.bc_use_entropy)
    bc_use_entropy_tuning = bool(cfg.algorithm.bc_use_entropy_tuning)
    ent_target_mult = float(cfg.algorithm.bc_ent_target_mult)

    for data in train_loader:
        obs = data['observations'].numpy()
        expert_action = data['actions'].numpy()

        # Normalize expert actions to [-1, 1] using bounds with safety margin
        expert_action = 2.0 * (expert_action - data_low) / (data_high - data_low) - 1.0

        # Convert to jax arrays
        obs = jnp.array(obs)
        expert_action = jnp.array(expert_action)

        (loss, metrics), grads = nnx.value_and_grad(compute_loss, has_aux=True)(
            actor, obs, expert_action, bc_use_entropy, bc_use_entropy_tuning, ent_target_mult, seed
        )
        opt_state.update(actor, grads)

        actor_loss_sum += float(loss)
        nll_loss_sum += float(metrics["nll_loss"])
        entropy_sum += float(metrics["entropy"])
        entropy_loss_sum += float(metrics["entropy_loss"])
        mean_sum += float(metrics["mean"])
        log_std_sum += float(metrics["log_std"])
        num_batches += 1

    return (
        actor_loss_sum / num_batches,
        nll_loss_sum / num_batches,
        entropy_sum / num_batches,
        entropy_loss_sum / num_batches,
        mean_sum / num_batches,
        log_std_sum / num_batches,
    )


def evaluate(cfg, val_loader, actor, data_low, data_high, seed):
    """Evaluate on validation set and return average metrics."""
    val_loss_sum = 0.0
    val_nll_loss = 0.0
    val_entropy = 0.0
    val_entropy_loss = 0.0
    val_mean = 0.0
    val_log_std = 0.0
    val_num_batches = 0

    bc_use_entropy = bool(cfg.algorithm.bc_use_entropy)
    bc_use_entropy_tuning = bool(cfg.algorithm.bc_use_entropy_tuning)
    ent_target_mult = float(cfg.algorithm.bc_ent_target_mult)

    for vdata in val_loader:
        obs = jnp.array(vdata['observations'].numpy())
        expert_action = vdata['actions'].numpy()

        # Normalize expert actions to [-1, 1]
        expert_action = 2.0 * (expert_action - data_low) / (data_high - data_low) - 1.0
        expert_action = jnp.array(expert_action)

        # Forward pass — reuse compute_loss for consistent val metrics
        val_loss, metrics = compute_loss(actor, obs, expert_action, bc_use_entropy, bc_use_entropy_tuning, ent_target_mult, seed)

        val_loss_sum += float(val_loss)
        val_nll_loss += float(metrics["nll_loss"])
        val_entropy += float(metrics["entropy"])
        val_entropy_loss += float(metrics["entropy_loss"])
        val_mean += float(metrics["mean"])
        val_log_std += float(metrics["log_std"])
        val_num_batches += 1

    return (
        val_loss_sum / val_num_batches,
        val_nll_loss / val_num_batches,
        val_entropy / val_num_batches,
        val_entropy_loss / val_num_batches,
        val_mean / val_num_batches,
        val_log_std / val_num_batches,
    )

def save_actor(actor, path):
    """Save JAX actor state using nnx.state + pickle."""
    _, state = nnx.split(actor)
    with open(path, 'wb') as f:
        pickle.dump(jax.device_get(state), f)
    print(f"  ✓ Saved JAX actor to {path}")

@hydra.main(version_base=None, config_path="../config/default", config_name="reppo_maniskill")
def main(cfg: OmegaConf):
    """Main training function."""
    OmegaConf.set_struct(cfg, False)

    EPOCHS = 100

    # Load demo parameters from config
    env_name = cfg.env.name
    batch_size = cfg.env.demo.batch_size
    demo_path = cfg.env.demo.demo_path
    max_episodes = cfg.env.demo.max_episodes
    filter_success = cfg.env.demo.filter_success

    seeds = get_pretrain_seeds(cfg)
    print(f"Running multi-seed pretraining with seeds: {seeds}")

    best_losses = []

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"\n{'#' * 70}")
        print(f"Run {run_idx}/{len(seeds)} | seed={seed}")
        print(f"{'#' * 70}")

        set_all_seeds(seed)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Initialize wandb per seed run
        wandb.init(
            config={**dict(cfg), "seed": seed},
            entity=cfg.logging.entity,
            project="bc-baseline",
            name=f"bc_pretrain_{cfg.env.name}_seed{seed}",
            mode=cfg.logging.mode,
            reinit=True,
        )

        # Load demonstrations (still uses torch DataLoader internally)
        train_loader, val_loader, n_obs, n_act, _, _ = load_demos_for_training(
            env_id=env_name,
            device='cpu',
            bsize=batch_size,
            demo_path=demo_path,
            max_episodes=max_episodes,
            filter_success=filter_success,
        )

        # Create JAX actor with seed-specific RNG
        rngs = nnx.Rngs(seed)
        actor = create_actor(n_obs, n_act, cfg, rngs)

        # Get action bounds from environment
        temp_env = gym.make(env_name, obs_mode="state_dict", control_mode=cfg.env.get('control_mode'))
        env_low = temp_env.action_space.low
        env_high = temp_env.action_space.high
        temp_env.close()
        print(f"True environment bounds: low={env_low}, high={env_high}")

        # Create optax optimizer
        optimizer = optax.adamw(learning_rate=float(cfg.algorithm.optimizer.learning_rate))
        opt_state = nnx.Optimizer(actor, optimizer, wrt=nnx.Param)

        # Compute dataset action stats for normalization bounds
        # Use raw trajectories (not DataLoader batches) to avoid drop_last=True excluding samples
        config = DemoConfig(device=torch.device("cpu"), filter_success_only=filter_success, cut_at_first_success=False)
        loader = ManiSkillDemoLoader(config, env_name)
        trajectories, _ = loader.load_demo_dataset(demo_path)
        all_actions = np.concatenate(
            [traj['actions'].numpy() for traj in trajectories], axis=0
        )

        data_low = all_actions.min(axis=0)
        data_high = all_actions.max(axis=0)

        # Add 10% safety margin to bounds
        margin = 0.1 * (data_high - data_low)
        low_with_margin = data_low - margin
        high_with_margin = data_high + margin

        # Ensure bounds are at least [-1, 1] in each dimension
        dataset_low = np.minimum(low_with_margin, -1.0)
        dataset_high = np.maximum(high_with_margin, 1.0)

        # Normalize actions for stats check
        normalized_actions = 2.0 * (all_actions - dataset_low) / (dataset_high - dataset_low) - 1.0

        print("\n=== DATASET ACTION STATS ===")
        print(f"Expert action min (raw): {all_actions.min():.6f}")
        print(f"Expert action max (raw): {all_actions.max():.6f}")
        print(f"Normalized action min: {normalized_actions.min():.6f}")
        print(f"Normalized action max: {normalized_actions.max():.6f}")
        print(f"Action normalization bounds (low): {dataset_low}")
        print(f"Action normalization bounds (high): {dataset_high}")
        print("================================\n")

        best_vloss = float('inf')
        train_losses = []
        val_losses = []

        for epoch in range(EPOCHS):
            print(f'\nEPOCH {epoch + 1}/{EPOCHS}')

            # Training
            avg_actor_loss, avg_nll_loss, avg_entropy, avg_entropy_loss, avg_mean, avg_log_std = train_one_epoch(
                cfg, train_loader, actor, opt_state, dataset_low, dataset_high, seed
            )
            train_losses.append(avg_actor_loss)

            # Validation
            avg_val_loss, v_nll_loss, v_entropy, v_entropy_loss, v_mean, v_log_std = evaluate(
                cfg, val_loader, actor, dataset_low, dataset_high, seed
            )
            val_losses.append(avg_val_loss)

            print(f'Loss: train={avg_actor_loss:.4f} | val={avg_val_loss:.4f}')

            # Log to wandb
            wandb.log({
                "seed": seed,
                "train/nll_loss": avg_nll_loss,
                "train/entropy": avg_entropy,
                "train/entropy_loss": avg_entropy_loss,
                "train/mean": avg_mean,
                "train/log_std": avg_log_std,
                "val/nll_loss": v_nll_loss,
                "val/entropy": v_entropy,
                "val/entropy_loss": v_entropy_loss,
                "val/mean": v_mean,
                "val/log_std": v_log_std,
                "epoch": epoch + 1,
            }, step=epoch + 1)

            # Save best actor based on validation loss
            if avg_val_loss < best_vloss:
                best_vloss = avg_val_loss
                model_dir = f'{env_name}/saved_models/{timestamp}_seed{seed}'
                os.makedirs(model_dir, exist_ok=True)
                model_path = f'{model_dir}/bc_model_actor_best.pkl'
                save_actor(actor, model_path)
                print(f'  Best val_loss={avg_val_loss:.4f}')

        best_losses.append(best_vloss)

        print(f"\n{'='*60}")
        print(f"Seed {seed} complete! Best validation loss: {best_vloss:.4f}")
        print(f"{'='*60}")

        # Save final loss plots
        plot_dir = f'{env_name}/loss_plots/{timestamp}_seed{seed}'
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label=f"Train Loss for {env_name}", color="blue")
        plt.plot(val_losses, label=f"Val Loss for {env_name}", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Actor Loss")
        plt.legend()
        plt.title(f'Losses for {env_name} (seed={seed})')
        plot_path = f'{plot_dir}/loss_plot.png'
        plt.savefig(plot_path)
        plt.close()

        print(f"Saved loss plots to {plot_path}")
        wandb.finish()

    mean_best = float(np.mean(best_losses))
    std_best = float(np.std(best_losses))

    print(f"\n{'*' * 70}")
    print(f"Multi-seed summary over {len(seeds)} runs")
    for s, loss in zip(seeds, best_losses):
        print(f"  seed={s}: best_val_loss={loss:.4f}")
    print(f"  mean(best_val_loss)={mean_best:.4f} | std={std_best:.4f}")
    print(f"{'*' * 70}")


if __name__ == "__main__":
    main()
