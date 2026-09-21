"""
Example: Use ProgramAsWeights with LangChain.

Replace LLM calls in your LangChain pipeline with local neural functions
for deterministic, fast, and free execution.

Usage:
    pip install programasweights langchain langchain-core
"""
from threading import Lock

from langchain_core.tools import tool
import programasweights as paw

_message_triage = paw.compile_and_load(
    "Classify if a message needs immediate attention or can wait. "
    "Return only 'immediate' or 'wait'."
)
_message_triage_lock = Lock()

_json_fixer = paw.compile_and_load(
    "Fix malformed JSON: repair missing quotes and trailing commas"
)
_json_fixer_lock = Lock()

_log_triage = paw.compile_and_load(
    "Extract only lines indicating errors or failures, ignore info and debug"
)
_log_triage_lock = Lock()


@tool
def triage_message(text: str) -> str:
    """Classify if a message requires immediate attention or can wait."""
    with _message_triage_lock:
        return _message_triage(text)


@tool
def fix_json(text: str) -> str:
    """Fix malformed JSON: repair missing quotes and trailing commas."""
    with _json_fixer_lock:
        return _json_fixer(text)


@tool
def triage_logs(text: str) -> str:
    """Extract only error lines from verbose logs."""
    with _log_triage_lock:
        return _log_triage(text)


if __name__ == "__main__":
    print("Tools registered:")
    for t in [triage_message, fix_json, triage_logs]:
        print(f"  {t.name}: {t.description}")

    print("\nTesting triage_message:")
    print(triage_message.invoke("Urgent: the server is down!"))

    print("\nTesting fix_json:")
    print(fix_json.invoke("{name: 'Alice', age: 30,}"))
