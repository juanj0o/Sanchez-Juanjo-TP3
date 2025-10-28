import numpy as np
import cupy as cp


def accuracy(y_true, y_pred):
    if isinstance(y_true, cp.ndarray):
        y_true = cp.asnumpy(y_true)
    if isinstance(y_pred, cp.ndarray):
        y_pred = cp.asnumpy(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=0)

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    return np.sum(y_true == y_pred) / len(y_true)


def confusion_matrix(y_true, y_pred, n_classes=None):
    if isinstance(y_true, cp.ndarray):
        y_true = cp.asnumpy(y_true)
    if isinstance(y_pred, cp.ndarray):
        y_pred = cp.asnumpy(y_pred)

    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=0)

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    if n_classes is None:
        n_classes = max(np.max(y_true), np.max(y_pred)) + 1

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1

    return cm


def precision_recall_per_class(y_true, y_pred, n_classes=None):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    n_classes = cm.shape[0]

    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)

    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


def f1_score_macro(y_true, y_pred, n_classes=None):
    precision, recall = precision_recall_per_class(y_true, y_pred, n_classes)
    n_classes = len(precision)
    f1_per_class = np.zeros(n_classes)

    for i in range(n_classes):
        if precision[i] + recall[i] > 0:
            f1_per_class[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1_per_class[i] = 0.0

    f1_macro = np.mean(f1_per_class)

    return f1_macro, f1_per_class


def evaluate_model(y_true, y_pred, n_classes=None, verbose=True):
    acc = accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    f1_macro, f1_per_class = f1_score_macro(y_true, y_pred, n_classes)
    precision, recall = precision_recall_per_class(y_true, y_pred, n_classes)

    metrics = {
        'accuracy': acc,
        'confusion_matrix': cm,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'precision_per_class': precision,
        'recall_per_class': recall
    }

    if verbose:
        print("=" * 60)
        print("metricas")
        print("=" * 60)
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"F1-Score Macro: {f1_macro:.4f}")
        print(f"Precision promedio: {np.mean(precision):.4f}")
        print(f"Recall promedio: {np.mean(recall):.4f}")
        print("=" * 60)

    return metrics


def plot_confusion_matrix(cm, class_names=None, normalize=False, title='Matriz de Confusi�n',
                         figsize=(12, 10), cmap='Blues'):
    import matplotlib.pyplot as plt

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='Clase Verdadera',
           xlabel='Clase Predicha')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if n_classes <= 20:
        thresh = cm.max() / 2.
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=8)

    fig.tight_layout()
    return fig, ax
