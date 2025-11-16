from sklearn.metrics import accuracy_score, f1_score

def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    try:
        f1 = f1_score(y_true, y_pred, average="binary")
    except Exception:
        f1 = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": acc, "f1": f1}
