# Extraction validation

Validated on 2026-10-27 with Python 3.13, Torch 2.11.0+cu128, MuJoCo/MuJoCo Warp 3.11.0, Warp 1.17.0 and official mjlab commit `b517e0c489139e7fcee95702cfb2b01931264985`. GPU checks used an NVIDIA RTX 4090. CPU tests ran in the same CUDA-capable environment.

## Code and packaging

- Ruff formatting/lint, ty and pyright pass.
- All 120 tests pass. They cover learning gradients, analytic heads, motion data, terminal transitions, sampling, replay, save/resume, two-rank CPU synchronization, evaluation, plots and migration compatibility. The only test warnings are upstream TorchScript deprecations.
- Wheel and source archive build successfully. Archive inspection confirms there is no mjlab source, checkpoint, replay, motion data, log or private asset configuration in either artifact.
- Every package module imports from an extracted wheel with Python's site processing disabled and no source checkout on `sys.path`; mjlab resolves from the independently installed dependency.

## Preserved checkpoint

The legacy `model_5000.pt` loads without rewriting it. Headless GPU playback and a full 40-motion evaluation complete without replay loading for inference.

The full training state restores with 4096 environments and replay capacity/size 1000. World-model, actor and critic optimizer update counts restore to 50000 / 5000 / 100000. One additional collection step advances real simulator transitions from 51331072 to 51335168. This check restores optimizer state but does not perform a full-size learner update.

Separate CPU CLI smoke runs perform one small training iteration, save a new checkpoint, resume it into a separate output directory and reach total iteration 2. The simulator integration test also checks actual parameter updates, sampler/replay continuity and normalizer restoration.

## Same-machine evaluation comparison

Original and extracted evaluators used the same legacy checkpoint, all 40 motions, seed 42, 40 parallel environments and the same RTX 4090.

| Metric | Original source | Standalone package |
| --- | ---: | ---: |
| Successful motions | 34/40 | 34/40 |
| Mean normalized return | 0.7853302584 | 0.7851239338 |
| Mean completion | 0.9141251616 | 0.9138291021 |

Every per-motion success/failure classification matches. Mean normalized return differs by 0.0002063; the largest per-motion normalized-return difference is 0.006255. These runs are not bitwise identical. This is a same-machine extraction comparison, not a claim that historical evaluations on other hardware reproduce exactly.

Local reports are under ignored `logs/eval/source_baseline/` and `logs/eval/migration_model_5000/`. Historical PPO/SuperMimic comparison plots were regenerated under `logs/train_plots/ppo_supermimic/` without changing their inputs.

## Preservation and limits

All 121 files in the archived checkpoint groups, motions and historical logs retain their original sizes and modification times. The original Git working tree remains clean. A separate, pre-existing `checkpoints/g1_supermimic/` training directory continued receiving event/checkpoint files during this work; it was left untouched by the migration, and was not treated as a static archive. Refresh the inventory before a future upload.

The native desktop viewer was not exercised on this headless host. Long production training and full-size learner updates after the preserved checkpoint were not run. Hugging Face upload/download and remote publication remain future work.
