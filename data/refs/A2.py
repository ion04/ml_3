# A2: прирост информации по Джини. Родитель (40; 10), левый (35; 2), правый (5; 8).
def gini(counts):
    n = sum(counts)
    return 1 - sum((c / n) ** 2 for c in counts)

parent, left, right = (40, 10), (35, 2), (5, 8)
n = sum(parent)
weighted = sum(left) / n * gini(left) + sum(right) / n * gini(right)
print(f"REFERENCE: {gini(parent) - weighted:.4f}")
