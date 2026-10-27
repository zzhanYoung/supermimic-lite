"""SuperMimic Table A.5 and explicitly documented reproduction defaults."""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl.config import RslRlBaseRunnerCfg


@dataclass
class AdamCfg:
  learning_rate: float = 2e-4
  betas: tuple[float, float] = (0.9, 0.999)
  eps: float = 1e-8
  weight_decay: float = 0.0
  max_grad_norm: float = 1.0


@dataclass
class WorldModelCfg:
  hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
  window_length: int = 8
  batch_size: int = 2048
  batches_per_update: int = 10
  optimizer: AdamCfg = field(default_factory=AdamCfg)


@dataclass
class ImaginationCfg:
  hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  latent_dim: int = 32
  horizon: int = 32
  batch_size: int = 4096
  elementwise_grad_clip: float = 0.1
  optimizer: AdamCfg = field(default_factory=lambda: AdamCfg(betas=(0.9, 0.95)))


@dataclass
class CriticCfg:
  hidden_dims: tuple[int, ...] = (256, 128)
  epochs: int = 5
  minibatches: int = 4
  optimizer: AdamCfg = field(default_factory=AdamCfg)


@dataclass
class SuperMimicRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "SuperMimicRunner"
  experiment_name: str = "g1_supermimic"
  logger: Literal["wandb", "tensorboard"] = "tensorboard"
  max_iterations: int = 320_000
  num_steps_per_env: int = 0
  """Must be zero: collection is scheduled by update_to_data_ratio."""
  initial_collection_steps: int = 32
  save_interval: int = 4000
  upload_model: bool = False
  load_checkpoint: str = r"model_\d+\.pt$"
  world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
  imagination: ImaginationCfg = field(default_factory=ImaginationCfg)
  critic: CriticCfg = field(default_factory=CriticCfg)
  replay_steps: int = 1000
  gamma: float = 0.99
  lam: float = 0.95
  action_noise: float = 0.1
  update_to_data_ratio: float = 0.4
  """Complete learner updates / simulator vector steps."""
  decay_iterations: int = 250_000
  final_lr_fraction: float = 0.03
  activation: Literal["elu", "silu", "tanh"] = "elu"
  compile: bool | None = None
  """Compile WM loss and imagination steps; None enables compilation on CUDA."""

  def validate(self) -> None:
    positive = (
      self.initial_collection_steps,
      self.replay_steps,
      self.save_interval,
      self.world_model.window_length,
      self.world_model.batch_size,
      self.world_model.batches_per_update,
      self.imagination.horizon,
      self.imagination.batch_size,
      self.imagination.latent_dim,
      self.critic.epochs,
      self.critic.minibatches,
      self.decay_iterations,
    )
    if any(x <= 0 for x in positive):
      raise ValueError("SuperMimic sizes and update counts must be positive.")
    if self.replay_steps < max(10, self.world_model.window_length):
      raise ValueError("Replay must fit observation history and a model window.")
    if self.num_steps_per_env != 0:
      raise ValueError(
        "Fixed num_steps_per_env is obsolete; set it to 0 and configure "
        "update_to_data_ratio (complete updates / simulator steps)."
      )
    if not 0 < self.update_to_data_ratio <= 1:
      raise ValueError("Expected 0 < update_to_data_ratio <= 1.")
    if self.initial_collection_steps < max(10, self.world_model.window_length):
      raise ValueError("Initial collection must fill history and a model window.")
    if not 0 < self.gamma < 1 or not 0 <= self.lam <= 1:
      raise ValueError("Expected 0 < gamma < 1 and 0 <= lambda <= 1.")
    if not 0 < self.final_lr_fraction <= 1 or self.action_noise < 0:
      raise ValueError("Invalid learning-rate floor or action noise.")
    for optimizer in (
      self.world_model.optimizer,
      self.imagination.optimizer,
      self.critic.optimizer,
    ):
      if min(optimizer.learning_rate, optimizer.eps, optimizer.max_grad_norm) <= 0:
        raise ValueError("Adam rates, epsilon and gradient limits must be positive.")
      if any(not 0 <= beta < 1 for beta in optimizer.betas):
        raise ValueError("Adam betas must be in [0, 1).")


def runner_config(values: dict) -> SuperMimicRunnerCfg:
  """Reconstruct nested dataclasses after the existing CLI's asdict boundary."""
  values = dict(values)
  for key, cls in (
    ("world_model", WorldModelCfg),
    ("imagination", ImaginationCfg),
    ("critic", CriticCfg),
  ):
    nested = dict(values[key])
    nested["optimizer"] = AdamCfg(**nested["optimizer"])
    values[key] = cls(**nested)
  cfg = SuperMimicRunnerCfg(**values)
  cfg.validate()
  return cfg
