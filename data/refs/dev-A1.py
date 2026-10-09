# dev-A1: AdaBoost, взвешенная ошибка слабого классификатора 0.2. Вес alpha = 0.5*ln((1-e)/e).
import math

e = 0.2
print(f"REFERENCE: {0.5 * math.log((1 - e) / e):.4f}")
