"""Lifecycle checks without starting, stopping, or unloading live services."""
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location('stack', Path(__file__).parents[1] / 'scripts/stack.py')
stack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stack)


def test_start_reuses_healthy_gateway(monkeypatch):
    monkeypatch.setattr(stack, 'health', lambda: {'gateway': 'phase2'})
    launch = Mock(side_effect=AssertionError('must not launch'))
    monkeypatch.setattr(stack.subprocess, 'Popen', launch)
    stack.start()
    launch.assert_not_called()


def test_stop_refuses_unmanaged_gateway(monkeypatch):
    monkeypatch.setattr(stack, 'busy_guard', lambda: None)
    monkeypatch.setattr(stack, 'owned_pid', lambda: None)
    monkeypatch.setattr(stack, 'health', lambda: {'gateway': 'phase2'})
    with pytest.raises(RuntimeError, match='started elsewhere'):
        stack.stop()


def test_busy_gateway_blocks_stop(monkeypatch):
    monkeypatch.setattr(stack, 'health', lambda: {'gateway': 'phase2'})
    monkeypatch.setattr(stack, 'api', lambda *a: {'active': [{'id': 'request'}]})
    with pytest.raises(RuntimeError, match='busy'):
        stack.stop()


def test_reused_pid_not_owned(monkeypatch, tmp_path):
    pidfile = tmp_path / 'proxy.pid'
    pidfile.write_text('12345')
    monkeypatch.setattr(stack, 'PIDFILE', pidfile)
    monkeypatch.setattr(stack.subprocess, 'run', lambda *a, **kw: Mock(returncode=0, stdout='/other/server.py'))
    assert stack.owned_pid() is None
