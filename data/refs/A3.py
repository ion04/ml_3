# A3: метрики по матрице ошибок TP=30, FP=10, FN=20, TN=940.
tp, fp, fn, tn = 30, 10, 20, 940
accuracy = (tp + tn) / (tp + fp + fn + tn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)
print(f"REFERENCE: {accuracy:.4f}; {precision:.4f}; {recall:.4f}; {f1:.4f}")
