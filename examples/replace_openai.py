"""
Example: Replace OpenAI API calls with ProgramAsWeights.

Before: $0.01-0.10 per call, internet required, rate limited, non-deterministic
After:  Free, local, instant, deterministic

Usage:
    pip install programasweights
    python replace_openai.py
"""
import programasweights as paw

# --- The old way: OpenAI API ---
#
# from openai import OpenAI
# client = OpenAI()
#
# def triage_openai(text: str) -> str:
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{
#             "role": "user",
#             "content": f"Classify if this requires immediate attention or can wait. "
#                        f"Reply with just 'immediate' or 'wait'.\n\n{text}"
#         }],
#         max_tokens=10,
#     )
#     return response.choices[0].message.content.strip().lower()

# --- The new way: ProgramAsWeights ---

triage = paw.function("email-triage")

test_messages = [
    "Urgent: production database is unresponsive, all writes failing",
    "Newsletter: team building event next Friday",
    "Action required: submit expense report by end of day",
    "FYI: new coffee machine in the break room",
]

print("Message Triage (runs locally, no API calls):\n")
for msg in test_messages:
    result = triage(msg)
    print(f"  [{result:>9}]  {msg}")
