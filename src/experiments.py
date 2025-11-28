# experiments.py
print("Importing libraries...")
import argparse
import numpy as np
import time
import spacy
import spacy_sentence_bert
from spacy.training import Example
from spacy.util import minibatch
from scipy.stats import pearsonr
from dataloader import load_ner_dataset, load_cls_dataset, load_sts_dataset
from data_formatter import format_ner_dataset, format_cls_dataset, format_sts_dataset
from utils import get_memory_usage, get_model_size, save_results, calculate_metric, get_cached_model_path
import random
print("Libraries imported.")



# Load models
def load_models(model_name: str):
    """Load spacy model

    Args:
        model_name (str): name of the model

    Returns:
        nlp: spaCy model
    """
    # First try to load directly (works if already installed or in sys.path)
    try:
        return spacy.load(model_name)
    except Exception as e:
        # Not installed: download then load
        spacy.cli.download(model_name)
        return spacy.load(model_name)




################################################################################
# NER Experiment
################################################################################

def run_ner_experiment(model_name, ner_dataset):
    """
    Run NER experiment
    
    Args:
        model_name (str): name of the spaCy model
        ner_dataset (list[dict]): NER dataset in spaCy format

    Returns:
        dict: evaluation results including precision, recall, f1, runtime, memory usage, and model size
    """
    print("Running NER experiment...")
    
    # Load model and start timer
    nlp_model = load_models("en_core_web_trf" if model_name == "transformer" else model_name)
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
    
    
    
    

################################################################################
# Similarity Experiment
################################################################################
    
def cosine_sim(vec1, vec2):
    """
    Compute cosine similarity between two vectors
    
    Args:
        vec1 (np.ndarray): first vector
        vec2 (np.ndarray): second vector
        
    Returns:
        float: cosine similarity
    """
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)

    
def run_similarity_experiment(model_name, sts_dataset):
    """
    Run similaity experiment

    Args:
        model_name (str): name of the spaCy model
        sts_dataset (list[dict]): similarity dataset in spaCy format

    Returns:
        dict: evaluation results including pearson correlation, runtime, memory usage, and model size
    """
    print("Running Similarity experiment...")
    nlp_model = spacy_sentence_bert.load_model('en_stsb_roberta_large') if model_name == "transformer" else load_models(model_name)
    model_path = get_cached_model_path(nlp_model, pipe_name="sentence_bert") if model_name == "transformer" else nlp_model.path

    # Compute similarities and measure runtime
    start_time = time.time()
    sims = []
    for d in sts_dataset:
        doc1 = nlp_model(d["sentence1"])
        doc2 = nlp_model(d["sentence2"])
        sims.append(doc1.similarity(doc2))
    runtime = time.time() - start_time

    # Get human scores
    human = [d["label"] for d in sts_dataset]

    return {
        "pearson": float(pearsonr(sims, human)[0]),
        "runtime_sec": runtime,
        "memory_mb": get_memory_usage(),
        "model_size_mb": get_model_size(model_path)
    }
    
    
 
 
    
  
################################################################################
# Classification Experiment
################################################################################ 
    
def build_textcat_pipeline(nlp_model, label_names):
    """
    Build a spaCy text classification pipeline with AG News labels.

    Args:
        nlp_model: spaCy Language model
        label_names: list[str]
            AG News labels like ["World", "Sports", "Business", "SciTech"]

    Returns:
        nlp: spaCy model with a configured TextCategorizer
    """
    # If pipeline already has textcat, remove it (avoid duplicates)
    if "textcat" in nlp_model.pipe_names:
        nlp_model.remove_pipe("textcat")
    textcat = nlp_model.add_pipe("textcat", last=True)
        
    # Register all AG News category labels
    for label in label_names:
        textcat.add_label(label)

    return nlp_model


def prepare_examples(cls_nlp, cls_dataset, label_names):
    """
    Convert data to Examples

    Args:
        cls_nlp (spacy.language.Language): spaCy Language model
        cls_dataset (list[dict]): classification dataset
        label_names (list[str]): list of label names for classification

    Returns:
        list[spacy.training.Example]: list of spaCy Example objects
    """
    examples = []
    for item in cls_dataset:
        text = item["text"]
        label = item["label"]
        doc = cls_nlp.make_doc(text)
        cats = {lbl: 0.0 for lbl in label_names}
        cats[label] = 1.0
        examples.append(Example.from_dict(doc, {"cats": cats}))   
    return examples


def train_categorizer(cls_nlp, train_examples, n_iter = 10, batch_size = 16):
    """
    Train base models

    Args:
        cls_nlp (_type_): _description_
        train_examples (_type_): _description_
        n_iter (int, optional): _description_. Defaults to 5.
        batch_size (int, optional): _description_. Defaults to 16.

    Returns:
        _type_: _description_
    """
    textcat = cls_nlp.get_pipe("textcat")
    textcat.initialize(get_examples=lambda: train_examples, nlp=cls_nlp)
    optimizer = cls_nlp.resume_training()
    
    for epoch in range(n_iter):
        random.shuffle(train_examples)
        epoch_loss = 0.0
        
        for batch in minibatch(train_examples, size=batch_size):
            losses = {}
            cls_nlp.update(batch, sgd=optimizer, losses=losses)
            epoch_loss += losses.get("textcat", 0.0)
        
        avg_loss = epoch_loss / (len(train_examples) / batch_size)
        print(f"[textcat] Epoch {epoch+1}/{n_iter}, loss: {avg_loss:.4f}")
    return cls_nlp
    
    

def run_classification_experiment(model_name, cls_dataset_train, cls_dataset_test, label_names):
    """
    Run classification experiment
    
    Args:
        model_name (str): name of the spaCy model
        cls_dataset_train (list[dict]): training classification dataset in spaCy format
        cls_dataset_test (list[dict]): testing classification dataset in spaCy format
        label_names (list[str]): list of label names for classification
        
    Returns:
        dict: evaluation results including precision, recall, f1, runtime, memory usage, and model size
    """
    print("Running Classification experiment...")

    # build and train textcat pipeline
    cls_nlp = load_models("./models/trf_textcat/model-best" if model_name == "transformer" else model_name)

    # convert datasets to spacy examples
    train_examples = prepare_examples(cls_nlp, cls_dataset_train, label_names)
    dev_examples = prepare_examples(cls_nlp, cls_dataset_test, label_names)
    dev_texts = [e.reference.text for e in dev_examples]
    dev_labels = [max(e.reference.cats, key=e.reference.cats.get) for e in dev_examples]
    
    # initialize and train classifier for base models
    if model_name != "transformer":
        print("Training text categorizer for base model...")
        cls_nlp = build_textcat_pipeline(cls_nlp, label_names) 
        cls_nlp = train_categorizer(cls_nlp, train_examples)
    
    # run experiment and measure runtime
    start_time = time.time()
    preds_raw = [cls_nlp(text).cats for text in dev_texts]
    runtime = time.time() - start_time

    # get predictions and calculate metrics
    preds = [max(cats, key=cats.get) for cats in preds_raw]
    metrics = calculate_metric(dev_labels, preds)

    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "runtime_sec": runtime,
        "memory_mb": get_memory_usage(),
        "model_size_mb": get_model_size(cls_nlp.path)
    }








if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True, choices=["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "transformer"], help="Name of the model to evaluate")
    args = argparser.parse_args()
    
    print("Loading datasets...")
    ner_ds = load_ner_dataset(split="train", limit=10000)
    sts_ds = load_sts_dataset(split="train", limit=10000)
    cls_ds_train = load_cls_dataset(split="train", limit=30000)
    cls_ds_test = load_cls_dataset(split="test", limit=10000)
    
    label_names = cls_ds_train.features["label"].names
    if label_names != cls_ds_test.features["label"].names:
        raise ValueError("Train and test classification datasets have different label names.")
    print(f"Classification labels: {label_names}")
    
    print("Formatting datasets...")
    formatted_ner_ds = format_ner_dataset(ner_ds)
    formatted_sts_ds = format_sts_dataset(sts_ds)
    formatted_cls_ds_train = format_cls_dataset(cls_ds_train)
    formatted_cls_ds_test = format_cls_dataset(cls_ds_test)

    print("Running experiments...")
    ner_result = run_ner_experiment(args.model, formatted_ner_ds)
    sts_result = run_similarity_experiment(args.model, formatted_sts_ds)
    cls_result = run_classification_experiment(args.model, 
                                               formatted_cls_ds_train, 
                                               formatted_cls_ds_test, 
                                               label_names)
    
    print("Saving results...")
    final_result = {
        "ner": ner_result,
        "similarity": sts_result,
        "classification": cls_result,
    } 
    save_results(f"./results/{args.model}_results.json", final_result)

    
