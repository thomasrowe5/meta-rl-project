import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to the CSV file
file_path = os.path.join("reports", "test_results.csv")

# Load the test results
df = pd.read_csv(file_path)

# Determine Pass/Fail: here we define Pass if Predicted_Action == 1
df["Pass"] = df["Predicted_Action"] == 1

# Count Pass vs Fail
summary = df["Pass"].value_counts()

# Plot bar chart
plt.figure(figsize=(6, 4))
summary.plot(kind="bar", color=["red", "green"])
plt.xticks(ticks=[0, 1], labels=["Fail", "Pass"], rotation=0)
plt.ylabel("Number of Scenarios")
plt.title("Test Results Pass/Fail")
plt.tight_layout()

# Save figure
os.makedirs("reports", exist_ok=True)
plt.savefig(os.path.join("reports", "test_results_chart.png"))
plt.show()

print("Visualization saved to reports/test_results_chart.png")
