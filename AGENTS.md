# AGENTS.md

## Cursor Cloud specific instructions

Project: **REPPO** (Relative Entropy Pathwise Policy Optimization) — a JAX + PyTorch
reinforcement-learning research codebase. There is a single "product": the training
entrypoint `src/train.py`, configured with Hydra (`config/`). Dependencies are managed
with `uv` (`pyproject.toml`); see `README.md` for the canonical run recipes.

### Environment / hardware
- The cloud VM is **CPU-only** (no GPU). JAX and PyTorch both run in CPU mode even though
  `pyproject.toml` pins `jax[cuda12]` — `jax.devices()` returns `CpuDevice` and everything
  still works, just slower. Do not treat the README's "CUDA 12 GPU required" note as a
  blocker for development/smoke testing.
- System build headers `python3.12-dev` + `build-essential` are required to `uv sync`
  (the ManiSkill dependency chain `mani-skill -> mplib -> toppra` compiles C/Cython
  extensions). These are installed at VM-setup time; the startup update script only runs
  `uv sync`.

### Running the trainer (non-obvious gotchas)
- The **default `python src/train.py` run is enormous** (`reppo_continuous` = MJX G1
  humanoid, `num_envs=1024`, `total_time_steps=50_000_000`). For a quick check, override
  `algorithm.num_envs`, `algorithm.num_eval`, and `algorithm.total_time_steps` down.
- In the current tree, the **only end-to-end-working env family is MJX
  `mujoco_playground` locomotion** (e.g. `Go1JoystickFlatTerrain`, `G1JoystickFlatTerrain`).
  The `LogWrapper` (`src/env_utils/jax_wrappers.py`) unconditionally reads
  `env_state.metrics["reward/tracking_lin_vel"]`, which only locomotion tasks provide.
  The `gymnax` / `gymnasium` / `minatar` / DMC configs currently error out (the gymnax
  runner also interpolates `${env.demo.demo_path}`, which those `config/env/*.yaml` files
  do not define), so prefer MJX locomotion for smoke tests.
- The scan/loop trainers **require `algorithm.data_type` to be `random` or `PER`** — the
  config default `online` raises `ValueError`. `PER` builds a Flashbax buffer with
  `device="gpu"`, so on the CPU VM use `algorithm.data_type=random`.
- Use `logging=wandb_offline` to avoid needing a W&B login. Hydra `chdir` is on, so run
  outputs land in `outputs/<date>/<time>/` (git-ignored).
- The first MJX run downloads `mujoco_menagerie` assets into `~/.cache` (needs network,
  cached afterwards).

Known-good CPU smoke command (finishes in ~1-2 min, mostly JIT compile):
```
uv run python src/train.py --config-name=reppo_continuous \
  env.name=Go1JoystickFlatTerrain logging=wandb_offline \
  algorithm.data_type=random algorithm.num_envs=8 \
  algorithm.num_eval=1 algorithm.total_time_steps=80 seed=0
```

### Lint / test
- Lint: `uvx ruff check src config` (Ruff config lives in `pyproject.toml`). The tree
  currently has pre-existing Ruff findings; do not treat them as regressions from your
  change.
- There is **no automated test framework** wired up. `test/langevin_preconditioned.py`
  and `src/test_bc*.py` are standalone research/eval scripts (some hard-code absolute
  paths), not a pytest suite.
