# A6: логистическая регрессия p = sigmoid(b0 + b1*h), b0=-4.1, b1=1.5.
import math

b0, b1 = -4.1, 1.5
p3 = 1 / (1 + math.exp(-(b0 + b1 * 3)))   # вероятность при h = 3
h50 = -b0 / b1                             # p = 0.5 там, где b0 + b1*h = 0
print(f"REFERENCE: {p3:.4f}; {h50:.4f}")
