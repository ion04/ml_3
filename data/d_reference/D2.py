# D2 (эталон): load_digits, PCA с сохранением 95% дисперсии + логистическая регрессия, accuracy на тесте.
# PCA обучается только на обучающей части (внутри Pipeline), число компонент выводится для проверки.
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95, random_state=42),
                     LogisticRegression(max_iter=5000))
pipe.fit(X_train, y_train)

print("компонент PCA:", pipe.named_steps["pca"].n_components_)
print(f"METRIC: {pipe.score(X_test, y_test):.4f}")
