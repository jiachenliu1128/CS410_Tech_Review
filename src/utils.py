import time
import psutil
import os
import json
from sklearn.metrics import precision_recall_fscore_support

def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024**2)

def get_model_size(model_path):
    size = 0
    for root, _, files in os.walk(model_path):
        for f in files:
            size += os.path.getsize(os.path.join(root, f))
    return size / (1024**2)

def save_results(out_path, results_dict):
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=4)
        
def calculate_metric(true_labels, pred_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
