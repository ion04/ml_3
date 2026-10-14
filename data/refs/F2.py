# F2 — изменённая версия A2 (прирост Джини). Пара P2 для H1.
# Впишите СВОИ числа: (объектов класса 0; объектов класса 1) в левом и правом потомке.
left = (28, 3)
right = (7, 12)

if None in (left, right):
    raise SystemExit("Впишите свои left и right, например left = (30, 4)")


def gini(n0, n1):
    p = n0 / (n0 + n1)
    return 1 - p ** 2 - (1 - p) ** 2


parent = (left[0] + right[0], left[1] + right[1])       # родитель = сумма потомков
n, nl, nr = sum(parent), sum(left), sum(right)
answer = gini(*parent) - (nl / n * gini(*left) + nr / n * gini(*right))
assert abs(answer - 0.1212) > 0.01, "Ответ почти совпадает с A2 (0.1212)"
print(f"# в условии: в узле {parent[0]} объектов класса 0 и {parent[1]} класса 1, левый {left}, правый {right}")
print(f"REFERENCE: {answer:.4f}")
