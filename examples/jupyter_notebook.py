"""
Example: Use ProgramAsWeights in a Jupyter Notebook for data cleaning.

Copy these cells into a Jupyter notebook.

Cell 1: Install
    !pip install programasweights pandas

Cell 2: Setup
"""
# Cell 2: Load programs
import programasweights as paw
import pandas as pd

# Load a program (downloads once, cached forever)
email_extractor = paw.function("programasweights/email-extractor")

# Cell 3: Process a DataFrame
data = pd.DataFrame({
    "id": [1, 2, 3],
    "text": [
        "Contact alice@company.com for sales inquiries",
        "Reach out to bob@example.org or carol@test.com",
        "No email addresses in this text",
    ]
})

# Apply the neural program to each row
data["emails"] = data["text"].apply(email_extractor)
print(data)

# Output:
#    id                                              text                              emails
# 0   1       Contact alice@company.com for sales inquiries              ["alice@company.com"]
# 1   2  Reach out to bob@example.org or carol@test.com  ["bob@example.org", "carol@test.com"]
# 2   3                No email addresses in this text                                     []


# Cell 4: Batch processing (more efficient)
texts = data["text"].tolist()
# Programs accept lists for batch processing
results = [email_extractor(t) for t in texts]
print(results)


# Cell 5: Compare with regex approach
"""
# Traditional regex approach (fragile, misses edge cases):
import re
pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
data["emails_regex"] = data["text"].apply(lambda x: re.findall(pattern, x))

# ProgramAsWeights approach (robust, handles edge cases):
data["emails_paw"] = data["text"].apply(email_extractor)

# Both produce same results, but PAW handles:
# - Obfuscated emails (alice [at] company.com)
# - Emails with unusual TLDs
# - Context-dependent extraction
"""
