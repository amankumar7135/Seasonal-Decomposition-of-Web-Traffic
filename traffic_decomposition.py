
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

plt.rcParams["figure.figsize"] = (12, 8)

df = pd.read_csv("web_traffic.csv", parse_dates=["date"])
df = df.sort_values("date").set_index("date")

full_dates = pd.date_range(df.index.min(), df.index.max(), freq="D")
df = df.reindex(full_dates)
df.index.name = "date"

df["traffic"] = df["traffic"].interpolate(method="time")

plt.plot(df.index, df["traffic"])
plt.title("Daily Website Traffic")
plt.xlabel("Date")
plt.ylabel("Traffic")
plt.show()

additive = seasonal_decompose(df["traffic"], model="additive", period=7)
additive.plot()
plt.show()

multiplicative = seasonal_decompose(df["traffic"], model="multiplicative", period=7)
multiplicative.plot()
plt.show()

residuals = additive.resid.dropna()
threshold = 2 * residuals.std()
anomalies = residuals[abs(residuals) > threshold]

print(anomalies)
