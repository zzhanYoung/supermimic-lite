"""Shared simulator test setup."""

import pytest
import warp as wp


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
  wp.config.quiet = True
