import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("metrics.csv")
plt.plot(df["round"], df["mta"], label="MTA")
plt.plot(df["round"], df["asr"], label="ASR")
plt.plot(df["round"], df["test_acc"], label="Test Acc")
plt.xlabel("Round")
plt.ylabel("Value")
plt.legend()
plt.show()