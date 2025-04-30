import numpy as np
from sklearn import metrics


def compute_youden_threshold(gt, prob):
    """
    Computes the optimal threshold using Youden's J statistic.

    Parameters:
    - gt: Ground truth binary labels (0 or 1)
    - prob: Predicted probabilities for the positive class

    Returns:
    - Optimal threshold based on Youden's J statistic
    """
    fpr, tpr, thresholds = metrics.roc_curve(gt, prob)
    J = tpr - fpr  # Youden's J statistic
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold


def calculate_classification_metrics(prob, gt, threshold=0.5):
    """
    Calculates classification metrics including AUC, accuracy, sensitivity, specificity, etc.

    Parameters:
    - gt: Ground truth binary labels (0 or 1)
    - prob: Predicted probabilities for the positive class
    - threshold: Threshold to convert probabilities into binary predictions

    Returns:
    - Dictionary containing various classification metrics
    """
    # Convert predicted probabilities to binary predictions based on the threshold
    pred = np.asarray(prob) > threshold

    # Compute ROC curve and AUC
    fpr_roc, tpr_roc, _ = metrics.roc_curve(gt, prob)
    auc_score = metrics.auc(fpr_roc, tpr_roc)

    # Compute accuracy
    accuracy = metrics.accuracy_score(gt, pred)

    # Compute confusion matrix components
    try:
        cm = metrics.confusion_matrix(gt, pred, labels=[0, 1])
    except Exception as e:
        print("Error in confusion_matrix:", e)
        cm = np.array([[0, 0], [0, 0]])

    try:
        tn, fp, fn, tp = cm.ravel()
    except ValueError as e:
        print(f"Warning: Could not unpack confusion matrix: {e}")
        tn = fp = fn = tp = 0

    # Compute sensitivity (recall) and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Compute false positive rate and false negative rate
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Compute positive predictive value (precision) and negative predictive value
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # Compute Precision-Recall AUC
    precision, recall, _ = metrics.precision_recall_curve(gt, prob)
    pr_auc_score = metrics.auc(recall, precision)

    # Return all metrics in a dictionary
    return {
        'AUC': auc_score,
        'ACC': accuracy,
        'SEN': sensitivity,
        'SPE': specificity,
        'FPR': false_positive_rate,
        'FNR': false_negative_rate,
        'PPV': ppv,
        'NPV': npv,
        'PR-AUC': pr_auc_score,
    }
