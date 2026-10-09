# A4: SVM, w=(3; 4), b=-2. Ширина зазора 2/||w|| и класс точки (1; 0).
import numpy as np

w, b, x = np.array([3.0, 4.0]), -2.0, np.array([1.0, 0.0])
width = 2 / np.linalg.norm(w)
score = w @ x + b                     # = 1: точка лежит ровно на границе зазора
label = 1 if score > 0 else -1
print(f"REFERENCE: {width:.4f}; {label}")
