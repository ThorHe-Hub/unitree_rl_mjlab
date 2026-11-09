"""Run one approved 30-trial bias batch. --prepare-only never starts a simulator."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import psutil
from heading_bias_batch import protocol, freeze, inspect_case, inspect_attempt, write_json
from run_heading_fixed_input import command_for, verify_identity, digest

PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT/'unitree_heading_results'


@contextmanager
def batch_lock(root):
    with (root/'runner.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def require_stopped(directory):
    """Also detect a child left alive if its launcher was forcibly closed."""
    for launch in directory.glob('attempt_*/launch_record.json'):
        record = json.loads(launch.read_text(encoding='utf-8'))
        pid, created = record.get('child_pid'), record.get('child_created')
        if pid is not None:
            try:
                process = psutil.Process(pid)
                if created is None or abs(process.create_time()-created)<.01:
                    raise RuntimeError(f'Previous child is still running (PID {pid}); wait for it to stop before resuming')
            except psutil.NoSuchProcess:
                pass
        expected = str(launch.parent.resolve())
        for process in psutil.process_iter(['pid', 'cmdline']):
            try:
                args = process.info['cmdline'] or []
                if '--output-dir' in args:
                    i = args.index('--output-dir')
                    if i+1<len(args) and os.path.normcase(os.path.abspath(args[i+1])) == os.path.normcase(expected):
                        raise RuntimeError(f'Previous simulation still running (PID {process.pid})')
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue


def stop_child(child):
    if child.poll() is not None:
        return
    try:
        parent = psutil.Process(child.pid)
        owned = parent.children(recursive=True)+[parent]
        for process in owned:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(owned, timeout=5)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=5)
    except psutil.NoSuchProcess:
        pass
    child.wait(timeout=10)


def mark_complete(directory, result):
    row = result[0]
    attempt = Path(row['trace_file']).parts[1]
    marker = directory/'completed.json'
    expected = dict(attempt=attempt, trace_sha256=row['sha256'], stop_reason=row['stop_reason'])
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != expected:
            raise ValueError('Final case marker changed')
    else:
        write_json(marker, expected)


def run_case(project, root, case, plan, resume):
    directory = root/case['case_id']
    if directory.exists():
        if not resume:
            raise RuntimeError('Case directory exists; use --resume to verify and continue')
        require_stopped(directory)
        result = inspect_case(directory, case, plan)
        if result is not None:
            mark_complete(directory, result)
            print(f'[SKIP VERIFIED] {case["case_id"]}: {result[0]["stop_reason"]}', flush=True)
            return result[0]
    directory.mkdir(exist_ok=True)
    attempts = sorted(directory.glob('attempt_*'))
    number = max([int(p.name.split('_')[-1]) for p in attempts] or [0])+1
    attempt = directory/f'attempt_{number:03d}'
    attempt.mkdir()
    command = command_for(project, case, attempt, plan['common'])
    record = dict(**case, command=command, status='launching',
                  started_utc=datetime.now(timezone.utc).isoformat())
    write_json(attempt/'launch_record.json', record)
    env = {**os.environ, 'GIT_PYTHON_REFRESH':'quiet', 'PYTHONIOENCODING':'utf-8', 'PYTHONUNBUFFERED':'1',
           'PYTHONPATH':str(project)+os.pathsep+os.environ.get('PYTHONPATH','')}
    child, started = None, time.monotonic()
    try:
        with (attempt/'console.txt').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            record.update(child_pid=child.pid, child_created=psutil.Process(child.pid).create_time(), status='running')
            write_json(attempt/'launch_record.json', record)
            for line in child.stdout:
                log.write(line)
                log.flush()
                if line.startswith(('[PROGRESS]', '[TRACE]')):
                    print(line.rstrip(), flush=True)
            record.update(returncode=child.wait(), status='process_ended')
    except BaseException as exc:
        if child is not None:
            stop_child(child)
        record.update(returncode=child.returncode if child is not None else None,
                      status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'infrastructure_error',
                      error=repr(exc))
        raise
    finally:
        record.update(wall_seconds=time.monotonic()-started, ended_utc=datetime.now(timezone.utc).isoformat())
        paths = list(attempt.glob('trace*.npz'))
        if len(paths)==1:
            record['trace_sha256'] = digest(paths[0])
        write_json(attempt/'run_record.json', record)
    result = inspect_attempt(attempt, case, plan)
    if result is None:
        tail = (attempt/'console.txt').read_text(encoding='utf-8')[-5000:]
        raise RuntimeError(f'Infrastructure failure retained in {attempt}. Fix the cause, then use --resume.\n{tail}')
    record['status'] = 'complete'
    write_json(attempt/'run_record.json', record)
    mark_complete(directory, result)
    return result[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bias', type=float, choices=(.02,.04), required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = (args.output_dir or RESULTS/f'bias_{int(round(args.bias*100)):03d}').resolve()
    root.mkdir(parents=True, exist_ok=True)
    with batch_lock(root):
        plan = freeze(PROJECT, root, protocol(PROJECT, args.bias))
        print(f'[PROTOCOL] {root/"protocol.json"}', flush=True)
        if args.prepare_only:
            print('[PREPARED ONLY] No simulation started.')
            return
        for i, case in enumerate(plan['cases'],1):
            verify_identity(PROJECT, plan)
            print(f'[CASE] {i}/30 bias={args.bias} {case["case_id"]}', flush=True)
            row = run_case(PROJECT, root, case, plan, args.resume)
            print(f'[END] {i}/30 steps={row["samples"]} reason={row["stop_reason"]}', flush=True)
        print(f'[COMPLETE] 30/30 final outcomes retained: {root}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[STOPPED] Current attempt retained. Repeat the same command with --resume.')
        raise SystemExit(130)
