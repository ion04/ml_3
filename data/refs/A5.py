# A5: МНК для y = b0 + b1*x по 5 точкам и прогноз в x = 6.
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)
b1 = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
b0 = y.mean() - b1 * x.mean()
print(f"REFERENCE: {b0:.4f}; {b1:.4f}; {b0 + b1 * 6:.4f}")
