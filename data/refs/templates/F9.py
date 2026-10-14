# F9 — k-means с лишними данными (тип F, distractor). Основа: ISLR (James et al.), упражнение на ручной k-means
# с 6 точками (гл. 12 во 2-м изд. / гл. 10 в 1-м, упр. 3). Там точки (1;4) (1;3) (0;4) (5;1) (6;2) (4;0) —
# ВОЗЬМИТЕ СВОИ координаты (исходные модели знают), задайте два начальных центра и добавьте в условие
# отвлекающие сведения (третий признак, который не используется; время замеров; силуэт прошлого запуска).
# Ответ: итоговые центры после сходимости, сначала центр, стартовавший из первого начального.
points = None     # например [(x, y), ...] — 6–8 точек
centers = None    # два начальных центра [(x, y), (x, y)]

if points is None or centers is None:
    raise SystemExit("Впишите свои points и centers")
for it in range(100):
    clusters = [[], []]
    for p in points:     # назначаем каждую точку ближайшему центру (при равенстве — первому)
        d = [(p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in centers]
        clusters[0 if d[0] <= d[1] else 1].append(p)
    assert all(clusters), "Один из кластеров пуст — выберите другие начальные центры"
    new = [(sum(p[0] for p in cl) / len(cl), sum(p[1] for p in cl) / len(cl)) for cl in clusters]
    if new == centers:
        break
    centers = new
print(f"# итераций до сходимости: {it}; кластеры: {clusters}")
print(f"REFERENCE: {centers[0][0]:.4f}; {centers[0][1]:.4f}; {centers[1][0]:.4f}; {centers[1][1]:.4f}")
