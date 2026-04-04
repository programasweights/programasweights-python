#!/usr/bin/env python3
"""
PAW-powered log monitor.

Watches a log file and uses a compiled PAW function to classify new output
as ALERT (significant events) or QUIET (routine progress). Compile a
domain-specific classifier once, then run this monitor forever.

Setup:
    pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/

    # Compile a classifier for your specific logs (do this once):
    python -c "
    import programasweights as paw
    p = paw.compile('''
    Classify log lines. Return ONLY one word: ALERT or QUIET.

    ALERT means: error, checkpoint, eval results, training finished.
    QUIET means: routine progress (loss, step counts, loading).

    Input: [step 100] loss=0.05 lr=0.0001
    Output: QUIET

    Input: [Checkpoint] Saved model at step 1000
    Output: ALERT

    Input: Traceback (most recent call last):
    Output: ALERT
    ''')
    print(p.id)
    " > .paw_monitor_id

Usage:
    python paw_monitor.py --log /path/to/training.log
    python paw_monitor.py --log /path/to/training.log --poll 10 --stall 3600
    python paw_monitor.py --log /path/to/training.log --stop-on-alert
"""

import argparse
import os
import sys
import time

import programasweights as paw

MAX_LINES_PER_CALL = 30


def _print(msg, **kwargs):
    print(msg, flush=True, **kwargs)


def _truncate_lines(lines, max_lines=MAX_LINES_PER_CALL):
    """Keep first and last lines, truncate middle if too long."""
    if len(lines) <= max_lines:
        return "\n".join(lines), 0
    keep_top = max_lines // 2
    keep_bot = max_lines - keep_top
    omitted = len(lines) - keep_top - keep_bot
    truncated = lines[:keep_top] + [f"[...{omitted} lines omitted...]"] + lines[-keep_bot:]
    return "\n".join(truncated), omitted


def _load_function(program_id=None):
    """Load PAW function by ID, or from .paw_monitor_id file."""
    if not program_id:
        id_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".paw_monitor_id")
        if not os.path.exists(id_path):
            _print(f"[monitor] No program ID provided and no .paw_monitor_id file found.")
            _print(f"[monitor] Compile a classifier first (see --help for setup instructions).")
            sys.exit(1)
        with open(id_path) as f:
            program_id = f.read().strip()

    _print(f"[monitor] Loading PAW function {program_id}...")
    fn = paw.function(program_id)
    _print(f"[monitor] Ready.")
    return fn


def monitor_log(log_path, fn, poll_interval=10, timeout=0, stall_timeout=3600,
                stop_on_alert=False):
    _print(f"[monitor] Watching: {log_path}")
    _print(f"[monitor] poll={poll_interval}s timeout={'unlimited' if timeout == 0 else f'{timeout}s'} stall={stall_timeout}s")

    if os.path.exists(log_path):
        with open(log_path) as f:
            f.seek(0, 2)
            file_pos = f.tell()
    else:
        file_pos = 0

    start_time = time.time()
    last_new_time = time.time()
    total_alerts = 0
    total_quiet = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if timeout > 0 and elapsed >= timeout:
                _print(f"[monitor] Timeout reached ({timeout}s). Exiting.")
                break

            if not os.path.exists(log_path):
                time.sleep(poll_interval)
                continue

            with open(log_path) as f:
                f.seek(file_pos)
                new_data = f.read()
                file_pos = f.tell()

            if not new_data:
                stall_elapsed = time.time() - last_new_time
                if stall_elapsed > stall_timeout:
                    _print(f"[monitor] STALL: No new output for {stall_elapsed:.0f}s")
                    last_new_time = time.time()
                time.sleep(poll_interval)
                continue

            last_new_time = time.time()
            raw_lines = [l.strip() for l in new_data.splitlines() if l.strip()]
            timestamp = time.strftime("%H:%M:%S")

            batch_text, n_omitted = _truncate_lines(raw_lines)
            verdict = fn(batch_text, max_tokens=1).strip().upper().split()[0]

            trunc_note = f" (truncated {n_omitted})" if n_omitted else ""

            if verdict == "ALERT":
                total_alerts += 1
                _print(f"[{timestamp}] ALERT | +{len(raw_lines)} lines{trunc_note}")
                for line in raw_lines:
                    _print(f"  | {line}")
                if stop_on_alert:
                    _print(f"[monitor] Alert after {elapsed:.0f}s. Exiting.")
                    break
            else:
                total_quiet += 1
                last_line = raw_lines[-1] if raw_lines else ""
                short = last_line if len(last_line) <= 80 else last_line[:77] + "..."
                _print(f"[{timestamp}] QUIET | +{len(raw_lines)} lines{trunc_note} | {short}")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        _print()

    _print(f"[monitor] Totals: {total_alerts} alerts, {total_quiet} quiet")


def main():
    parser = argparse.ArgumentParser(
        description="PAW-powered log monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup:
  1. Compile a classifier for your logs (once):
     python -c "import programasweights as paw; p = paw.compile('...your spec...'); print(p.id)" > .paw_monitor_id

  2. Run the monitor:
     python paw_monitor.py --log /path/to/training.log
""",
    )
    parser.add_argument("--log", required=True, help="Path to the log file to watch")
    parser.add_argument("--program", default=None, help="PAW program ID (reads .paw_monitor_id if not set)")
    parser.add_argument("--poll", type=int, default=10, help="Poll interval in seconds (default: 10)")
    parser.add_argument("--timeout", type=int, default=0, help="Total timeout in seconds (0=unlimited)")
    parser.add_argument("--stall", type=int, default=3600, help="Alert if no output for N seconds (default: 3600)")
    parser.add_argument("--stop-on-alert", action="store_true", help="Exit after first ALERT")
    args = parser.parse_args()

    fn = _load_function(args.program)
    monitor_log(args.log, fn, poll_interval=args.poll, timeout=args.timeout,
                stall_timeout=args.stall, stop_on_alert=args.stop_on_alert)


if __name__ == "__main__":
    main()
