# A8: PCA по ковариационной матрице [[2, 1], [1, 2]]: собственные значения и доля дисперсии PC1.
import numpy as np

cov = np.array([[2.0, 1.0], [1.0, 2.0]])
eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
print(f"REFERENCE: {eig[0]:.4f}; {eig[1]:.4f}; {eig[0] / eig.sum():.4f}")
