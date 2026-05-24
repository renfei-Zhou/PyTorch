import pandas as pd
import json
import matplotlib.pyplot as plt

### 8. Compare model results and train time
with open("model_0_results.json", "r", encoding="utf-8") as f:
    model_0_results = json.load(f)

with open("model_1_results.json", "r", encoding="utf-8") as f:
    model_1_results = json.load(f)

with open("model_2_results.json", "r", encoding="utf-8") as f:
    model_2_results = json.load(f)

compare_results = pd.DataFrame([model_0_results,
                                model_1_results,
                                model_2_results])

print(f"compare results:\n", compare_results.to_string())

# Visualize our model results
compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
plt.xlabel("accuracy (%)")
plt.ylabel("model")
plt.show()

debug=1