import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/engineered_train.csv")

engine_id = 1

engine_df = df[df["engine_id"] == engine_id]

plt.figure(figsize=(10,6))
plt.plot(engine_df["cycle"], engine_df["RUL"], label="True RUL")
plt.xlabel("Cycle")
plt.ylabel("Remaining Useful Life")
plt.title(f"Engine {engine_id} Degradation Curve")
plt.legend()
plt.tight_layout()
plt.savefig("models/rul_curve_engine1.png")
plt.close()

print("RUL curve saved.")