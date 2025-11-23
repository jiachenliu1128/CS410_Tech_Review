# experiments.py

import argparse
import time, os, psutil
import numpy as np
from dataloader import load_ner_dataset, load_cls_dataset, load_sts_dataset
import spacy
from spacy.training import Example
from scipy.stats import pearsonr
from utils import get_memory_usage, get_model_size, save_results, calculate_metric



# Load models
def load_models(model_name):
    nlp_model = spacy.load(model_name) 
    return nlp_model



# NER Experiment
def run_ner_experiment(nlp_model, ner_dataset):
    start_time = time.time()
    
    # create spacy examples
    spacy_examples = [
        Example.from_dict(nlp_model(d["text"]), {"entities": d["entities"]})
        for d in ner_dataset
    ]
    
    # Calculate scores and runtime
    scores = nlp_model.evaluate(spacy_examples)
    runtime = time.time() - start_time

    return {
        "precision": scores["ents_p"],
        "recall": scores["ents_r"],
        "f1": scores["ents_f"],
        "runtime_sec": runtime,
        "memory_mb": get_memory_usage(),
        "model_size_mb": get_model_size(nlp_model.path)
    }
    
    
    
    
# Classification Experiment
def run_classification_experiment(nlp_model, cls_dataset):
    # prepare data
    texts = [d["text"] for d in cls_dataset]
    labels = [d["label"] for d in cls_dataset]
    
    # run experiment and measure runtime
    start_time = time.time()
    preds_raw = [nlp_model(text).cats for text in texts]
    runtime = time.time() - start_time

    # get predictions and calculate metrics
    preds = [max(cats, key=cats.get) for cats in preds_raw]
    metrics = calculate_metric(labels, preds)

    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "runtime_sec": runtime,
        "memory_mb": get_memory_usage(),
        "model_size_mb": get_model_size(nlp_model.path)
    }
    
   
    
    
# Similarity Experiment
def run_similarity_experiment(nlp_model, sts_dataset):
    # Compute similarities and measure runtime
    start_time = time.time()
    sims = []
    for d in sts_dataset:
        doc1 = nlp_model(d["sent1"])
        doc2 = nlp_model(d["sent2"])
        sims.append(doc1.similarity(doc2))
    runtime = time.time() - start_time

    # Get human scores
    human = [d["score"] for d in sts_dataset]

    return {
        "pearson": float(pearsonr(sims, human)[0]),
        "runtime_sec": runtime,
        "memory_mb": get_memory_usage(),
        "model_size_mb": get_model_size(nlp_model.path)
    }




if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True, help="Name of the model to evaluate")
    args = argparser.parse_args()
    
    nlp_model = load_models(args.model)
    ner_ds = load_ner_dataset()
    cls_ds = load_cls_dataset()
    sts_ds = load_sts_dataset()

    run_similarity_experiment(nlp_model, sts_ds)
    run_classification_experiment(nlp_model, cls_ds)
    run_ner_experiment(nlp_model, ner_ds)
    
    
