"""
Example: Use ProgramAsWeights in a Jupyter Notebook for data processing.

Copy these cells into a Jupyter notebook.

Usage:
    pip install programasweights pandas
"""

# Cell 1: Setup
import programasweights as paw
import pandas as pd

triage = paw.function("email-triage")

# Cell 2: Process a DataFrame
data = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "message": [
        "Urgent: production server is down, customers are affected!",
        "Newsletter: team picnic next Saturday",
        "Action needed: submit your quarterly report by EOD",
        "FYI: new parking policy starts next month",
    ],
})

data["urgency"] = data["message"].apply(triage)

print(data[["id", "message", "urgency"]])
#    id                                            message    urgency
# 0   1  Urgent: production server is down, customers...  immediate
# 1   2           Newsletter: team picnic next Saturday       wait
# 2   3  Action needed: submit your quarterly report b...  immediate
# 3   4    FYI: new parking policy starts next month       wait

# Cell 3: Compile a custom function for log triage
log_triage = paw.function(
    paw.compile(
        "Extract only lines indicating errors or failures from this log, "
        "ignore info and debug lines"
    ).id
)

logs = """[INFO] Server started on port 8080
[DEBUG] Loading config...
[ERROR] Connection refused: database timeout
[INFO] Retrying...
[ERROR] Max retries exceeded"""

print(log_triage(logs))
# [ERROR] Connection refused: database timeout
# [ERROR] Max retries exceeded
