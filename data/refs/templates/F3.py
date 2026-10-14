# F3 — изменённая версия A3 (матрица ошибок). Пара P3 для H1. Спрашиваем то же: accuracy, precision, recall, F1.
TP, FP, FN, TN = None, None, None, None    # в A3: 30, 10, 20, 940

if None in (TP, FP, FN, TN):
    raise SystemExit("Впишите свои TP, FP, FN, TN")
acc = (TP + TN) / (TP + FP + FN + TN)
prec, rec = TP / (TP + FP), TP / (TP + FN)
f1 = 2 * prec * rec / (prec + rec)
assert (round(prec, 3), round(rec, 3)) != (0.75, 0.6), "precision/recall совпадают с A3"
print(f"REFERENCE: {acc:.4f}; {prec:.4f}; {rec:.4f}; {f1:.4f}")
