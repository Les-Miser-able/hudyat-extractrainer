"""Classification metrics."""
import numpy as np

def evaluation(y_true, probabilities, classes):
    predicted = probabilities.argmax(axis=1)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    np.add.at(cm, (y_true, predicted), 1)
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros(len(classes)), where=cm.sum(axis=0)>0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros(len(classes)), where=cm.sum(axis=1)>0)
    f1 = np.divide(2*precision*recall, precision+recall,
                   out=np.zeros(len(classes)), where=(precision+recall)>0)
    return {"accuracy": float(np.mean(predicted == y_true)), "macro_f1": float(f1.mean()),
            "confusion_matrix_rows_true_columns_predicted": cm.tolist(), "classes": classes,
            "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": int(cm[i].sum())}
                          for i, label in enumerate(classes)}}

