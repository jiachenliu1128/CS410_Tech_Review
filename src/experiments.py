# experiments.py

import argparse
import Example
import time, os, psutil
import numpy as np
from dataloader import load_ner_dataset, load_cls_dataset, load_sts_dataset
import spacy
from utils import measure_runtime, get_memory_usage, get_model_size, save_results

# Load models
def load_models(model_name):
    nlp_model = spacy.load(model_name) 
    return nlp_model

# NER Experiment
def run_ner_experiment(nlp_model, ner_dataset):
    # measuring runtime for NER recognition
    results, runtime = measure_runtime(lambda: [nlp_model(d["text"]) for d in ner_dataset])
    
    # convert examples for spaCy evaluation
    spacy_examples = [
        Example.from_dict(nlp_model.make_doc(ex["text"]), {"entities": ex["entities"]})
        for ex in ner_dataset
    ]

    scores = nlp_model.evaluate(spacy_examples)

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
    ...
    
# NER Experiment
def run_similarity_experiment(nlp_model, sts_dataset):
    ...

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