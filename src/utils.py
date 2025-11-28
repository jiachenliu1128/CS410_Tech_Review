import time
import psutil
import os
import json
from sklearn.metrics import precision_recall_fscore_support
from transformers.utils import cached_file   

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
    
def get_cached_model_path(nlp_model, pipe_name="transformer"):
    model_name = nlp_model.get_pipe(pipe_name).model._first_module().auto_model.name_or_path
    config_path = cached_file(model_name, "config.json")
    return config_path.replace("config.json", "")
