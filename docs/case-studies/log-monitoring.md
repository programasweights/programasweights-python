# Event-Driven Log Monitoring

Long-running processes -- training runs, deployments, data pipelines -- produce thousands of log lines. You want to know when something important happens (checkpoint saved, error, completion) without watching the terminal. PAW lets you compile a classifier that runs locally and alerts on the lines that matter.

**Full tool:** [examples/paw_monitor.py](https://github.com/programasweights/programasweights-python/blob/main/examples/paw_monitor.py) on GitHub.

## How we built it

### Attempt 1: Keyword matching

The obvious first approach: grep for "error", "fail", "complete".

**Result:** Too noisy. "error" matches "error_count=0" (routine metric). "complete" matches "batch complete" (every 10 seconds). Missing important lines like "Traceback" or "CUDA out of memory" that don't contain the keywords.

**Lesson:** The whole reason to use PAW is that keyword matching can't express "important enough to interrupt me."

### Attempt 2: One-shot description

```
Classify if this log line is important enough to alert on. Return ALERT or QUIET.
```

**Result:** Too vague. The model alerted on everything that looked unusual, including routine debug output. "Important" means different things in different contexts.

**Lesson:** Don't rely on abstract instructions. The model needs concrete examples of what ALERT and QUIET look like in YOUR specific logs.

### Attempt 3: Example-based spec

```
Classify log lines. Return ONLY one word: ALERT or QUIET.

Input: [step 100] loss=0.05 lr=0.0001
Output: QUIET

Input: [Checkpoint] Saved model at step 1000
Output: ALERT

Input: Traceback (most recent call last):
Output: ALERT

Input: Training complete. Final loss: 0.11
Output: ALERT
```

**Result:** This worked. The model learned the boundary between routine metrics (QUIET) and significant events (ALERT) from the examples. Adding 3-5 examples from actual logs was the key.

**Lesson:** Include examples from your actual data. 3-5 representative examples consistently outperform prose-only descriptions.

### Refinement: Developer feedback

A developer using PAW for monitoring diffusion model training shared their experience. Key insights:

- **Positive instructions work better than negative ones.** "Output ALERT only if X, Y, or Z" is more reliable than "Don't alert on routine output."
- **The spec should match the monitoring context.** Different training phases produce different log patterns. Examples should cover the specific domain.
- **Stall detection needs a separate mechanism.** PAW classifies what it sees -- it can't detect the absence of output. The monitoring tool needs a timer for "no new output in N minutes."

## The solution

### The spec

```python
program = paw.compile("""
Classify log lines. Return ONLY one word: ALERT or QUIET.

Input: [step 100] loss=0.05 lr=0.0001
Output: QUIET

Input: [Checkpoint] Saved model at step 1000
Output: ALERT

Input: Traceback (most recent call last):
Output: ALERT

Input: Training complete. Final loss: 0.11
Output: ALERT
""")

fn = paw.function(program.id)
```

Compile once, save the program ID, reuse forever. The function runs locally with no internet after the first download.

### The monitoring loop

```python
import time

fn = paw.function("your-program-id")

last_size = 0
while True:
    size = os.path.getsize(log_file)
    if size > last_size:
        with open(log_file) as f:
            f.seek(last_size)
            new_text = f.read()
        last_size = size

        # Truncate to last ~1000 chars to fit context window
        chunk = new_text[-1000:] if len(new_text) > 1000 else new_text
        result = fn(chunk)
        if result.strip() == "ALERT":
            send_notification(chunk)

    time.sleep(10)
```

### Full tool

The complete [paw_monitor.py](https://github.com/programasweights/programasweights-python/blob/main/examples/paw_monitor.py) adds:

- **File watching** with `seek()` to only process new content
- **Input truncation** to fit the ~2048 token context window
- **Stall detection** -- alerts if no new output for a configurable timeout
- **`--focus` and `--ignore`** flags to guide what the classifier pays attention to
- **`--local` flag** to run entirely offline after first compile
- **`--json` output** for integration with other tools

## Adapting this for your use case

1. **Collect 5-10 example log lines** from your actual process -- a mix of routine output and important events
2. **Write a spec** with `Input: ... Output: ALERT/QUIET` pairs
3. **Test with real logs** -- pipe a log file through the classifier and check which lines it flags
4. **Iterate** -- if it alerts too much, add more QUIET examples. If it misses things, add ALERT examples.
5. **Save the program ID** -- compile once, monitor forever

## Takeaways

- **Examples beat descriptions** for teaching the model your specific log patterns.
- **Compile once, run forever.** The compiled function is cached locally and needs no internet.
- **PAW classifies what it sees** -- combine with a timer for stall detection.
- **Iterate with real data.** The first spec is rarely perfect. Test, check failures, adjust examples.
