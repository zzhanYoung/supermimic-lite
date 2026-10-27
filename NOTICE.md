# Provenance

SuperMimic Lite was extracted from the local `supermimic-lite` implementation at revision `ecdc405` ("refine details"). Existing Apache-2.0 licensing is retained.

Task configuration, motion utilities and the environment step implementation derive in part from mjlab, maintained by the MuJoCo Lab contributors under Apache-2.0. The framework and robot assets are installed from official upstream; they are not vendored here.

The independent package structure was informed by `supermimic_reference/supermimic_mjlab`. The learner remains the lite implementation.

`typings/mujoco` contains generated MuJoCo 3.11.0 Python API declarations and regeneration scripts retained from the source repository. They are development support, not a simulator implementation.

Checkpoints, motion data and logs are excluded from source distribution. Record their provenance and distribution terms alongside any future asset release.
