"""Längere Keep-Alive-Zeit für uvicorn (core/server_tuning.py)."""

import uvicorn

from core.server_tuning import KEEP_ALIVE_SECONDS, extend_keep_alive


def test_running_server_config_gets_longer_keep_alive():
    config = uvicorn.Config(app=None)
    assert (
        config.timeout_keep_alive == 5
    )  # Standard von uvicorn, zu kurz hinter dem Proxy
    assert extend_keep_alive() >= 1
    assert config.timeout_keep_alive == KEEP_ALIVE_SECONDS > 600
