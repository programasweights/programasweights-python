"""
Example: Use ProgramAsWeights with LangChain.

Replace LLM calls in your LangChain pipeline with local neural functions
for deterministic, fast, and free execution.

Usage:
    pip install programasweights langchain langchain-core
"""
from langchain_core.tools import tool
import programasweights as paw


@tool
def triage_message(text: str) -> str:
    """Classify if a message requires immediate attention or can wait."""
    fn = paw.function("email-triage")
    return fn(text)


@tool
def fix_json(text: str) -> str:
    """Fix malformed JSON: repair missing quotes and trailing commas."""
    fn = paw.function(
        paw.compile(
            "Fix malformed JSON: repair missing quotes and trailing commas"
        ).id
    )
    return fn(text)


@tool
def triage_logs(text: str) -> str:
    """Extract only error lines from verbose logs."""
    fn = paw.function(
        paw.compile(
            "Extract only lines indicating errors or failures, ignore info and debug"
        ).id
    )
    return fn(text)


if __name__ == "__main__":
    print("Tools registered:")
    for t in [triage_message, fix_json, triage_logs]:
        print(f"  {t.name}: {t.description}")

    print("\nTesting triage_message:")
    print(triage_message.invoke("Urgent: the server is down!"))

    print("\nTesting fix_json:")
    print(fix_json.invoke("{name: 'Alice', age: 30,}"))
