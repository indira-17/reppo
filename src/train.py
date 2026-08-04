import hydra
import jax
import logging
import time
import wandb
import torch
from omegaconf import DictConfig, OmegaConf

from src.algorithms import envs, utils
from src.common import InitFn, LearnerFn, PolicyFn
from src.cfg_utils import fix_cfg

logging.basicConfig(level=logging.INFO)


@hydra.main(
    version_base=None,
    config_path="../config/default",
    config_name="reppo_continuous.yaml",
)
def main(cfg: DictConfig):
    cfg = fix_cfg(cfg)
    OmegaConf.resolve(cfg)
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    seed = int(cfg.seed)

    # Seed non-JAX libraries as well.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    demo_cfg = cfg.env.get("demo", {})
    demo_path = demo_cfg.get("demo_path", None)
    filter_success = cfg.env.get("filter_success", demo_cfg.get("filter_success", True))
    cut_at_first_success = cfg.env.get(
        "cut_at_first_success", demo_cfg.get("cut_at_first_success", True)
    )

    bc_indicator = cfg.algorithm.get("bc_indicator", False)
    data_type = cfg.algorithm.get("data_type", "online")

    base_run_name = (
        f"bc-reppo-{cfg.env.name}-retrace"
        if cfg.algorithm.bc_indicator
        else f"reppo-{cfg.env.name}"
    )

    # All five Slurm array tasks receive the same group.
    run_group = os.environ.get(
        "WANDB_RUN_GROUP",
        f"{base_run_name}-five-seed",
    )

    wandb.init(
        mode=cfg.logging.mode,
        project=cfg.logging.project,
        entity=cfg.logging.entity,
        tags=cfg.tags,
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"{run_name}-seed{cfg.seed}",
        group=os.environ.get("WANDB_RUN_GROUP"),
        job_type="train",
        save_code=True,
    )

    key = jax.random.PRNGKey(cfg.seed)

    if bc_indicator:
        from src.maniskill_utils.maniskill_dataloader_shabnam import DemoConfig, ManiSkillDemoLoader
        if demo_path is None:
            raise ValueError("bc_indicator=True requires cfg.env.demo.demo_path.")
        logging.info(f"Loading dataset from {demo_path}")
        demo_loader_cfg = DemoConfig(
            device=torch.device("cpu"),
            filter_success_only=filter_success,
            cut_at_first_success=cut_at_first_success,
        )
        loader = ManiSkillDemoLoader(demo_loader_cfg, cfg.env.name)
        trajectories, _ = loader.load_demo_dataset(demo_path)
        n_obs_dataset = trajectories[0]["observations"].shape[1]
        logging.info(f"Dataset observation dimension: {n_obs_dataset}")
        env_setup = envs.make_env(cfg, n_obs_dataset=n_obs_dataset)
        obs_space = env_setup.observation_space
    else:
        env_setup = envs.make_env(cfg)
        obs_space = env_setup.observation_space

    init_fn: InitFn = hydra.utils.call(cfg.algorithm.init)(
        cfg=cfg,
        observation_space=obs_space,
        action_space=env_setup.action_space,
    )
    learner_fn: LearnerFn = hydra.utils.call(cfg.algorithm.learner)(
        cfg=cfg,
        observation_space=obs_space,
        action_space=env_setup.action_space,
    )
    policy_fn: PolicyFn = hydra.utils.call(cfg.algorithm.policy)(
        cfg=cfg,
        action_space=env_setup.action_space,
        observation_space=obs_space,
    )

    rollout_fn = hydra.utils.call(cfg.runner.rollout_fn)(
        env_setup.env,
        demo_path=demo_path,
        data_type=data_type,
        filter_success=filter_success,
        cut_at_first_success=cut_at_first_success
    )
    eval_fn = hydra.utils.call(cfg.runner.eval_fn)(
        env_setup.eval_env,
        demo_path=demo_path,
        data_type=data_type,
        filter_success=filter_success,
        cut_at_first_success=cut_at_first_success,
    )

    make_train_fn = hydra.utils.call(cfg.runner.train_fn)
    train_fn = make_train_fn(
        env=(env_setup.env, env_setup.eval_env),
        init_fn=init_fn,
        learner_fn=learner_fn,
        policy_fn=policy_fn,
        rollout_fn=rollout_fn,
        eval_fn=eval_fn,
        log_callback=utils.make_log_callback(),
        demo_path=demo_path,
        filter_success=filter_success,
        cut_at_first_success=cut_at_first_success,
        wandb_run=run,
        critic_offline_warmup_iters=cfg.algorithm.get("critic_offline_warmup_iters", 0),
        data_type=data_type,
        max_buffer_size=cfg.algorithm.get("max_buffer_size", 1_000_000),
        per_alpha=cfg.algorithm.get("per_alpha", 0.6),
        per_beta=cfg.algorithm.get("per_beta", 0.4),
    )

    start = time.perf_counter()
    _, metrics = train_fn(key)
    jax.block_until_ready(metrics)
    duration = time.perf_counter() - start
    logging.info(f"Training took {duration:.2f} seconds.")
    wandb.finish()


if __name__ == "__main__":
    main()