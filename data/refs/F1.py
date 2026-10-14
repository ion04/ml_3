# F1 — изменённая версия A1 (Байес, датчик). Пара P1 для H1.
# Впишите СВОИ числа вместо None, запустите: .venv/bin/python data/refs/templates/F1.py
# Если всё в порядке, скопируйте файл в data/refs/F1.py, а REFERENCE — в reference задачи F1.
p_d = 0.02        # доля узлов с утечкой
sens = 0.90       # P(сработал | утечка)
fpr = 0.04        # P(сработал | нет утечки)
n_fires = 1       # сколько раз датчик сработал (независимые срабатывания); 2 = усложнённое условие

if None in (p_d, sens, fpr):
    raise SystemExit("Впишите свои числа p_d, sens, fpr в начало файла")
like_d, like_nd = sens ** n_fires, fpr ** n_fires          # правдоподобие n срабатываний подряд
answer = like_d * p_d / (like_d * p_d + like_nd * (1 - p_d))
original = 0.95 * 0.01 / (0.95 * 0.01 + 0.05 * 0.99)       # ответ A1 = 0.161
assert abs(answer - original) > 0.05, "Ответ почти совпадает с A1 — для H1 изменение должно менять ответ"
print(f"REFERENCE: {answer:.4f}")
