import time
import psutil
import os
import json

def measure_runtime(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    return result, time.time() - start

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
