# experiments.py

import time, os, psutil
import numpy as np
from datasets import load_dataset
import spacy

# 1. Load models (Basic version & Transformer version)
def load_models():
    nlp_base = spacy.load("en_core_web_md")      # 标准 spaCy
    nlp_trf = spacy.load("en_core_web_trf")      # spaCy-transformers
    return nlp_base, nlp_trf

# 2. Data loading functions: NER / Classification / Similarity
def load_ner_dataset():
    ...
def load_cls_dataset():
    ...
def load_sts_dataset():
    ...

# 3. Utility functions: Measure runtime, memory, convert to Doc, etc.
def measure_runtime_and_memory(nlp, texts):
    ...
def get_doc_vectors(nlp, texts):
    ...

# 4. Three experiments: NER / Classification / Similarity
def run_similarity_experiment(nlp_base, nlp_trf, sts_dataset):
    ...
def run_classification_experiment(nlp_base, nlp_trf, cls_dataset):
    ...
def run_ner_experiment(nlp_base, nlp_trf, ner_dataset):
    ...

if __name__ == "__main__":
    nlp_base, nlp_trf = load_models()
    ner_ds = load_ner_dataset()
    cls_ds = load_cls_dataset()
    sts_ds = load_sts_dataset()

    run_similarity_experiment(nlp_base, nlp_trf, sts_ds)
    run_classification_experiment(nlp_base, nlp_trf, cls_ds)
    run_ner_experiment(nlp_base, nlp_trf, ner_ds)
