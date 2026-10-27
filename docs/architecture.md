# Dependency boundary

The distribution is `supermimic-lite`; its import namespace is `supermimic`.

| Component | Owner |
| --- | --- |
| Simulation, scene, entity, managers, sensors, G1 assets, viewers | Official external mjlab |
| World model, actor, critic, replay, runner | `supermimic.learning` |
| Motion loading, return sampler, rewards, observations, terminations | `supermimic.mdp`, `supermimic.data` |
| Task configuration | `supermimic.tasks` |
| Kinematics and terminal-transition sequencing | `supermimic.envs.SuperMimicEnv` |
| CLI | `supermimic.cli` |
| Legacy checkpoint symbols | `supermimic.compat` |

`SuperMimicEnv` subclasses the official environment. Its step override updates kinematics before rewards/termination and preserves the true terminal transition before the runner's manual reset. Simulation, managers and reset implementation remain upstream. Simulator collection/resume tests cover this boundary.

No files are installed into the `mjlab` namespace. The CLI constructs local tasks without modifying mjlab's registry. RSL-RL's wrapper and logger are reused without exposing a PPO trainer.

Algorithms follow original lite revision `ecdc405`. The reference `supermimic_mjlab` informs the independent-package structure only; its learner and reset strategy were not substituted.

A per-load unpickler translates legacy `mjlab.tasks.tracking.mdp` symbols. Canonical signatures normalize namespace changes while retaining dataset and numeric configuration checks. YAML snapshots are parsed as inert data and restored against known configuration templates. Source checkpoints are never rewritten.

When upgrading mjlab, change its exact commit, lockfile and doctor check together. Re-run simulator tests and checkpoint acceptance checks before accepting that revision.
