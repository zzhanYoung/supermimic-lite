"""Two CPU ranks exercise the same reductions used by the GPU learner."""

import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed
from torch import nn
from torch.multiprocessing.spawn import spawn

from supermimic.learning.algorithm import optimizer_step
from supermimic.learning.config import AdamCfg, SuperMimicRunnerCfg
from supermimic.learning.models import RunningNorm
from supermimic.learning.runner import SuperMimicRunner
from supermimic.mdp.return_sampler import ReturnSampler, ReturnSamplerCfg

dist: Any = torch.distributed


def _distributed_worker(rank: int, directory: str) -> None:
  torch.set_num_threads(1)
  dist.init_process_group(
    "gloo",
    init_method=f"file://{directory}/rendezvous",
    rank=rank,
    world_size=2,
    timeout=timedelta(seconds=45),
  )
  try:
    runner = SuperMimicRunner.__new__(SuperMimicRunner)
    runner.cfg = SuperMimicRunnerCfg()
    runner.collection_remainder = 0.0
    schedule = [runner.next_collection_steps() for _ in range(10)]
    assert schedule == [2, 3] * 5
    sampler = ReturnSampler(
      torch.tensor([101]), 0.02, 0.092, ReturnSamplerCfg(), seed=rank
    )
    # Unequal observation counts distinguish a sum/count reduction from
    # incorrectly averaging each rank's local means.
    returns = torch.tensor([2.0]) if rank == 0 else torch.tensor([4.0, 6.0])
    sampler.observe(torch.zeros(len(returns), dtype=torch.long), returns)
    sampler.commit()
    torch.testing.assert_close(sampler.returns, torch.tensor([0.4, 0.0]))
    # One rank finishing no episodes still participates in the same history EMA.
    history = sampler.history.clone()
    if rank == 0:
      sampler.observe(torch.tensor([1]), torch.tensor([5.0]))
    sampler.commit()
    torch.testing.assert_close(sampler.returns, torch.tensor([0.4, 0.5]))
    difficulty = torch.softmax(-25 * sampler.returns / sampler.g_max, 0)
    torch.testing.assert_close(sampler.history, history.lerp(difficulty, 0.01))
    history = sampler.history.clone()
    sampler.commit()
    torch.testing.assert_close(sampler.history, history)
    normalizer = RunningNorm(1)
    normalizer.update(returns[:, None])
    torch.testing.assert_close(normalizer.mean, torch.tensor([4.0]))
    torch.testing.assert_close(normalizer.variance, torch.tensor([8 / 3]))
    assert normalizer.count == 3
    module = nn.Linear(2, 1, bias=False)
    nn.init.zeros_(module.weight)
    inputs = torch.tensor([[10.0, 0.0] if rank == 0 else [-6.0, 2.0]])
    optimizer = torch.optim.Adam(module.parameters(), lr=2e-4)
    norm = optimizer_step(module(inputs).sum(), module, optimizer, AdamCfg())
    assert norm == pytest.approx(math.sqrt(5))
    torch.testing.assert_close(
      module.weight.grad, torch.tensor([[2.0, 1.0]]) / math.sqrt(5)
    )
    _, frames = sampler.sample(32)
    torch.save(
      {
        "probabilities": sampler.probabilities(),
        "history": sampler.history,
        "frames": frames,
        "schedule": schedule,
      },
      Path(directory) / f"rank{rank}.pt",
    )
  finally:
    dist.destroy_process_group()


@pytest.mark.slow
def test_two_rank_returns_moments_and_gradients(tmp_path: Path) -> None:
  spawn(_distributed_worker, args=(str(tmp_path),), nprocs=2)
  ranks = [torch.load(tmp_path / f"rank{rank}.pt") for rank in range(2)]
  assert torch.equal(ranks[0]["probabilities"], ranks[1]["probabilities"])
  assert torch.equal(ranks[0]["history"], ranks[1]["history"])
  assert not torch.equal(ranks[0]["frames"], ranks[1]["frames"])
  assert ranks[0]["schedule"] == ranks[1]["schedule"]
