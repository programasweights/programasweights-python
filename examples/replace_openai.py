"""
Example: Replace OpenAI API calls with ProgramAsWeights.

Before: $0.01-0.10 per call, internet required, rate limited
After:  Free, local, instant, deterministic

Usage:
    pip install programasweights
    python replace_openai.py
"""
import programasweights as paw


# ============================================================
# BEFORE: Using OpenAI (expensive, slow, unreliable)
# ============================================================

def extract_emails_openai(text: str) -> str:
    """Old way: OpenAI API call."""
    import openai  # $$$
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Extract all email addresses from this text as a JSON list:\n\n{text}"
        }],
        temperature=0,
    )
    return response.choices[0].message.content
    # Cost: ~$0.03 per call
    # Latency: 1-5 seconds
    # Requires: internet, API key, rate limit handling


# ============================================================
# AFTER: Using ProgramAsWeights (free, fast, local)
# ============================================================

extract_emails = paw.function("programasweights/email-extractor")
# Cost: $0
# Latency: 10-50ms
# Requires: nothing (runs locally)


# ============================================================
# Side-by-side comparison
# ============================================================

if __name__ == "__main__":
    test_text = """
    Hi team,
    
    Please send the final report to alice.smith@company.com and 
    cc john_doe123@vendor.co.uk. For billing questions contact 
    billing@services.example.org.
    
    Thanks,
    coordinator@company.com
    """
    
    print("ProgramAsWeights result:")
    result = extract_emails(test_text)
    print(f"  {result}")
    print()
    print("Benefits:")
    print("  - Free (no API costs)")
    print("  - Fast (10-50ms vs 1-5s)")
    print("  - Offline (no internet needed)")
    print("  - Deterministic (same input = same output)")
    print("  - No rate limits")
    print("  - No API key management")
