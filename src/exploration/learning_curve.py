"""
Learning curve analysis for RUL model.

Analyzes performance improvement with increasing data size.
"""
import matplotlib.pyplot as plt

results = model.evals_result()

train_rmse = results["validation_0"]["rmse"]

plt.plot(train_rmse)
plt.title("Validation RMSE over boosting rounds")
plt.xlabel("Iterations")
plt.ylabel("RMSE")
plt.show()

if __name__ == "__main__":
    print("Running evaluation module...")