# experiments.py
print("Importing libraries...")
import argparse
from pathlib import Path
import time
import spacy
from spacy.training import Example
from spacy.util import minibatch
from scipy.stats import pearsonr
from dataloader import load_ner_dataset, load_cls_dataset, load_sts_dataset
from data_formatter import format_ner_dataset, format_cls_dataset, format_sts_dataset
from utils import get_memory_usage, get_model_size, save_results, calculate_metric
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



# NER Experiment
def run_ner_experiment(model_name, ner_dataset):
    """Run NER experiment

    Args:
        model_name (str): name of the spaCy model
        ner_dataset (list[dict]): NER dataset in spaCy format

    Returns:
        dict: evaluation results including precision, recall, f1, runtime, memory usage, and model size
    """
    print("Running NER experiment...")
    
    # Load model and start timer
    nlp_model = load_models(model_name)
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

    # For transformer models
    # if "trf" in nlp_model.meta["name"]:
    #     config = {"model": {"@architectures": "spacy-transformers.TransformerModel"}}
    #     textcat = nlp_model.add_pipe("textcat", config=config)
    # else:
    #     textcat = nlp_model.add_pipe("textcat")
    textcat = nlp_model.add_pipe("textcat", last=True)
        
    # Register all AG News category labels
    for label in label_names:
        textcat.add_label(label)

    return nlp_model




def initializing_textcat(nlp_model, texts, labels, label_names):
    textcat = nlp_model.get_pipe("textcat")
    train_length = int(len(texts) * 0.2)
    
    # Use a small subset of data to initialize model dimensions
    init_examples = []
    for text, label in zip(texts[:train_length], labels[:train_length]): 
        doc = nlp_model.make_doc(text)
        cats = {lbl: 0.0 for lbl in label_names}
        cats[label] = 1.0
        init_examples.append(Example.from_dict(doc, {"cats": cats}))

    # Initialize ONLY the textcat component (do NOT call nlp_model.initialize())
    textcat.initialize(get_examples=lambda: init_examples, nlp=nlp_model)
    
    


def train_categorizer(cls_nlp, train_examples, n_iter = 50, batch_size = 16):
    textcat = cls_nlp.get_pipe("textcat")
    textcat.initialize(get_examples=lambda: train_examples, nlp=cls_nlp)
    for epoch in range(n_iter):
        random.shuffle(train_examples)
        losses = {}
        for batch in minibatch(train_examples, size=batch_size):
            cls_nlp.update(batch, losses=losses)
        print(f"[textcat] Epoch {epoch+1}/{n_iter}, loss: {losses.get('textcat', 0.0):.4f}")
    return cls_nlp
    
    
    

    
# Classification Experiment
def run_classification_experiment(model_name, cls_dataset, label_names):
    """Run classification experiment
    Args:
        model_name (str): name of the spaCy model
        cls_dataset (list[dict]): classification dataset in spaCy format
        label_names (list[str]): list of class labels
    Returns:
        dict: evaluation results including precision, recall, f1, runtime, memory usage, and model size
    """
    print("Running Classification experiment...")
    
    # build and train textcat pipeline
    cls_nlp = load_models(model_name)
    cls_nlp = build_textcat_pipeline(cls_nlp, label_names)
    
    # prepare examples
    random.shuffle(cls_dataset)
    examples = []
    for item in cls_dataset:
        text = item["text"]
        label = item["label"]
        doc = cls_nlp.make_doc(text)
        cats = {lbl: 0.0 for lbl in label_names}
        cats[label] = 1.0
        examples.append(Example.from_dict(doc, {"cats": cats}))   

    # 80% train / 20% validation
    split = int(len(examples) * 0.8)
    train_examples = examples[:split]
    dev_examples = examples[split:]
    dev_texts = [e.reference.text for e in dev_examples]
    dev_labels = [max(e.reference.cats, key=e.reference.cats.get) for e in dev_examples]
    
    # train classifier
    print("Training text categorizer...")
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
    
   
    
    
# Similarity Experiment
def run_similarity_experiment(model_name, sts_dataset):
    """Run similaity experiment

    Args:
        model_name (str): name of the spaCy model
        sts_dataset (list[dict]): similarity dataset in spaCy format

    Returns:
        dict: evaluation results including pearson correlation, runtime, memory usage, and model size
    """
    print("Running Similarity experiment...")
    nlp_model = load_models(model_name)

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
        "model_size_mb": get_model_size(nlp_model.path)
    }




if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, required=True, help="Name of the model to evaluate")
    args = argparser.parse_args()
    
    print("Loading datasets...")
    ner_ds = load_ner_dataset()
    cls_ds = load_cls_dataset()
    sts_ds = load_sts_dataset()
    
    print("Formatting datasets...")
    formatted_ner_ds = format_ner_dataset(ner_ds)
    formatted_cls_ds = format_cls_dataset(cls_ds)
    formatted_sts_ds = format_sts_dataset(sts_ds)

    print("Running experiments...")
    ner_result = run_ner_experiment(args.model, formatted_ner_ds)
    cls_result = run_classification_experiment(args.model, formatted_cls_ds, label_names=cls_ds.features["label"].names)
    sts_result = run_similarity_experiment(args.model, formatted_sts_ds)
    
    print("Saving results...")
    final_result = {
        "ner": ner_result,
        "classification": cls_result,
        "similarity": sts_result
    } 
    save_results(f"./data/{args.model}_results.json", final_result)

    
