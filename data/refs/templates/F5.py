# F5 — изменённая версия A5 (МНК, y = b0 + b1·x). Пара P5 для H1. Ответ: b0, b1, прогноз при x_new.
points = None     # в A5: [(1, 2), (2, 4), (3, 5), (4, 4), (5, 5)]
x_new = None      # в A5: 6

if points is None or x_new is None:
    raise SystemExit("Впишите свои points и x_new")
n = len(points)
mx = sum(p[0] for p in points) / n
my = sum(p[1] for p in points) / n
b1 = sum((p[0] - mx) * (p[1] - my) for p in points) / sum((p[0] - mx) ** 2 for p in points)
b0 = my - b1 * mx
assert (round(b0, 3), round(b1, 3)) != (2.2, 0.6), "Коэффициенты совпадают с A5"
print(f"REFERENCE: {b0:.4f}; {b1:.4f}; {b0 + b1 * x_new:.4f}")
