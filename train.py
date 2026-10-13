import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import DATASETS

def train(key, cfg):
    df = pd.read_csv(cfg["csv"]).drop_duplicates()
    df = df.dropna(subset=[cfg["target"]])
    X = df.drop(columns=[cfg["target"]] + cfg.get("drop", []))
    y = df[cfg["target"]].astype(str)

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000))])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)

    print(f"\n=== {cfg['title']} ===")
    print(f"Rows: {len(df)} | Features: {X.shape[1]} | Test accuracy: {acc:.4f}")
    print(classification_report(y_te, pred))
    print("Confusion matrix:\n", confusion_matrix(y_te, pred))

    classes = list(model.classes_)
    bundle = {
        "model": model,
        "title": cfg["title"],
        "features": X.columns.tolist(),
        "numeric": {c: {"min": float(X[c].min()), "max": float(X[c].max()),
                        "median": float(X[c].median())} for c in numeric},
        "categorical": {c: sorted(X[c].dropna().astype(str).unique().tolist())
                        for c in categorical},
        "class_names": [cfg.get("labels", {}).get(c, c) for c in classes],
        "accuracy": float(acc),
    }
    joblib.dump(bundle, f"models/{key}.joblib")

if __name__ == "__main__":
    for k, c in DATASETS.items():
        train(k, c)
