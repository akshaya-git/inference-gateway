#!/usr/bin/env python3
"""Lifecycle for the local oMLX app and its inference gateway."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.inference-stack'
PIDFILE = STATE / 'proxy.pid'
LOG = STATE / 'proxy.log'
UPSTREAM = os.environ.get('OMLX_UPSTREAM', 'http://127.0.0.1:8000').rstrip('/')
GATEWAY = 'http://127.0.0.1:9000'


def api(base, path, method='GET', timeout=5):
    headers = {}
    key = os.environ.get('OMLX_API_KEY')
    if base == UPSTREAM and key:
        headers['Authorization'] = f'Bearer {key}'
    request = urllib.request.Request(base + path, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def health():
    try:
        result = api(GATEWAY, '/health')
    except (OSError, ValueError):
        return None
    if result.get('gateway') != 'phase2':
        raise RuntimeError('Port 9000 belongs to another service; refusing to manage it.')
    return result


def busy_guard():
    if health() is None:
        return
    data = api(GATEWAY, '/metrics')  # Fail closed if metrics cannot be read.
    jobs = data.get('benchmarks', {}).get('jobs', [])
    if data.get('active') or any(j.get('status') in ('queued', 'running') for j in jobs):
        raise RuntimeError('Gateway is busy. Wait for requests and benchmarks to finish.')


def owned_pid():
    try:
        pid = int(PIDFILE.read_text())
    except (OSError, ValueError):
        return None
    if pid <= 1:
        return None
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    # Never signal a reused PID, another server, or a process we cannot inspect.
    if result.returncode or str(ROOT / 'proxy.py') not in result.stdout:
        return None
    return pid


def start():
    if health() is not None:
        print(f'Gateway already running: {GATEWAY}')
        return
    try:
        api(UPSTREAM, '/admin/api/models')
    except urllib.error.HTTPError:
        raise RuntimeError('oMLX rejected the status request. Check OMLX_API_KEY.') from None
    except (OSError, ValueError):
        if UPSTREAM != 'http://127.0.0.1:8000':
            raise RuntimeError(f'Start your configured oMLX server at {UPSTREAM}.') from None
        subprocess.run(['open', '-a', '/Applications/oMLX.app'], check=True)
        print('Waiting for the oMLX app on port 8000...', flush=True)
        deadline = time.monotonic() + 90
        while True:
            try:
                api(UPSTREAM, '/admin/api/models')
                break
            except (OSError, ValueError):
                if time.monotonic() >= deadline:
                    raise RuntimeError('oMLX is not ready. Enable Auto Start in the app and check port 8000.')
                time.sleep(1)
    if owned_pid():
        raise RuntimeError('The managed proxy exists but is not healthy; inspect its log before restarting.')
    env = os.environ.copy()
    env.update(PROXY_HOST='127.0.0.1', PROXY_PORT='9000', OMLX_UPSTREAM=UPSTREAM,
               INFERENCE_STACK_STATE=str(STATE), PYTHONUNBUFFERED='1')
    with LOG.open('a') as output:
        child = subprocess.Popen([str(ROOT / '.venv/bin/python'), str(ROOT / 'proxy.py')],
                                 cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    PIDFILE.write_text(str(child.pid))
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if child.poll() is not None:
            PIDFILE.unlink(missing_ok=True)
            raise RuntimeError(f'Gateway exited. See {LOG}')
        if health() is not None:
            print(f'Gateway ready: {GATEWAY}\nPi endpoint: {GATEWAY}/v1')
            return
        time.sleep(0.25)
    child.terminate()
    try:
        child.wait(timeout=10)
        PIDFILE.unlink(missing_ok=True)
    except subprocess.TimeoutExpired:
        pass
    raise RuntimeError(f'Gateway startup timed out. See {LOG}')


def stop():
    busy_guard()
    pid = owned_pid()
    if pid is None:
        if health() is not None:
            raise RuntimeError('Gateway was started elsewhere. Stop it in its original terminal first.')
        print('Managed gateway is not running.')
        return
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if owned_pid() is None:
            PIDFILE.unlink(missing_ok=True)
            print('Gateway stopped. oMLX and its loaded models remain available.')
            return
        time.sleep(0.25)
    raise RuntimeError('Gateway did not exit within 15 seconds; no forced kill was issued.')


def status():
    print(f'Gateway: {"running" if health() else "not reachable"} ({GATEWAY})')
    try:
        models = api(UPSTREAM, '/admin/api/models')['models']
    except (OSError, ValueError, KeyError) as exc:
        print(f'oMLX: not reachable ({UPSTREAM}): {exc}')
        return
    print(f'oMLX: running ({UPSTREAM})')
    for model in models:
        state = 'loading' if model.get('is_loading') else 'loaded' if model.get('loaded') else 'unloaded'
        print(f'  {state:8} {model["id"]}')


def main():
    parser = argparse.ArgumentParser(description='Start oMLX + gateway; stop/restart affects the gateway only.')
    parser.add_argument('command', choices=['start', 'stop', 'restart', 'proxy-restart', 'status', 'logs',
                                           'moe-on', 'dense-on'])
    parser.add_argument('log', nargs='?', choices=['proxy', 'omlx', 'all'], default='proxy')
    args = parser.parse_args()
    if args.command == 'status':
        status()
        return
    if args.command == 'logs':
        paths = [str(LOG)] if args.log == 'proxy' else [str(Path.home() / '.omlx/logs/server.log')]
        if args.log == 'all':
            paths.append(str(LOG))
        os.execvp('tail', ['tail', '-n', '60', '-F', *paths])
    STATE.mkdir(exist_ok=True)
    with (STATE / 'lifecycle.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another stack command is in progress.') from None
        if args.command in ('stop', 'restart', 'proxy-restart'):
            stop()
        if args.command in ('start', 'restart', 'proxy-restart'):
            start()
        if args.command in ('moe-on', 'dense-on'):
            start()
            route = 'moe' if args.command == 'moe-on' else 'dense'
            api(GATEWAY, f'/control/model/{route}', method='POST', timeout=300)
            status()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
