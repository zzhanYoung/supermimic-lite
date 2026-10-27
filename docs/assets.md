# Assets and future Hugging Face migration

Git contains code, tests, configuration examples, documentation and the lockfile. Local paths, weights, replay, motion files, TensorBoard events and generated reports are ignored.

The workspace retains the original ten SuperMimic checkpoint groups (iterations 500–5000), replay shards, sampler snapshots, historical PPO run, evaluation reports and 40 LAFAN1 motion files in place. The local TOML file references them. PPO is comparison data only.

| Use | Required assets |
| --- | --- |
| Play/evaluate | `model_N.pt`, `params/env.yaml`, `params/agent.yaml`, compatible motions |
| Resume | Inference assets, all `model_N.rankR.replay.pt` shards, matching dataset and rank count |
| Plot | TensorBoard events and saved parameters, or exported scalar snapshots |
| Archive | All checkpoints, replay, `.sampling.npz`, configs, motions, logs and reports |

The iteration-5000 main checkpoint is about 209 MiB; its replay shard is about 6.57 GiB. Replay can be an optional download for inference users. Training snapshots preserve sampler state; `.sampling.npz` also records inspectable sampling distributions.

`supermimic data inventory --output logs/asset-inventory.json` records logical roots, relative names, file sizes and checkpoint relationships from `protected_roots`. It contains no absolute host paths. This is an inventory, not a checksum verifier or uploader.

A future Hugging Face snapshot can preserve the layout:

```text
checkpoints/<run>/model_N.pt
checkpoints/<run>/model_N.rank0.replay.pt
checkpoints/<run>/model_N.sampling.npz
checkpoints/<run>/params/{env,agent}.yaml
checkpoints/<run>/events.out.tfevents.*
motions/<dataset>/*.npz
logs/eval/<evaluation>/...
logs/train_plots/<comparison>/...
asset-inventory.json
```

Before upload, choose the hosting repository and immutable revision, add content checksums and dataset provenance, and select default replay downloads. A downloaded snapshot can use the same directory layout; only the asset configuration then changes. This extraction does not create a Hugging Face repository or upload files.
