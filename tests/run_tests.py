import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from model.model import MetaRLAgent
from scenarios.generate_scenarios import generate_test_scenarios

# Initialize agent
agent = MetaRLAgent()

# Generate test scenarios
scenarios = generate_test_scenarios(10)  # 10 dummy scenarios

# Run tests
results = []
for i, state in enumerate(scenarios):
    action = agent.predict(state)
    results.append({
        "Scenario_ID": i,
        "State": state.tolist(),
        "Predicted_Action": action
    })

print("Results length:", len(results))  # Should print 10

# Save results in reports/ inside project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
reports_path = os.path.join(project_root, "reports")
os.makedirs(reports_path, exist_ok=True)

csv_path = os.path.join(reports_path, "test_results.csv")
df = pd.DataFrame(results)
df.to_csv(csv_path, index=False)
print(f"Test results saved to {csv_path}")
