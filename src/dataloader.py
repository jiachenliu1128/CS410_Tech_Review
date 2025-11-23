from datasets import load_dataset

# 1. NER Dataset: CoNLL-2003
def load_ner_dataset(split="validation", limit=200):
    # loading the conll2003 dataset
    # limiting the size
    # Example data from check_datasets.py:
    # {'tokens': ['CRICKET', '-', 'LEICESTERSHIRE', ...], 'ner_tags': [0, 0, 3, ...]}
    # need 'trust_remote_code=True' because we are using an older dataset script.
    dataset = load_dataset("conll2003", split=split, trust_remote_code=True)
    dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset

# 2. Classification Dataset: AG News
def load_cls_dataset(split="test", limit=1000):
    # using AG News for the classification task
    # Example data:
    # {'text': "Fears for T N pension after talks...", 'label': 2}
    # label 2 corresponds to 'Business'
    dataset = load_dataset("ag_news", split=split)
    dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset

# 3. Similarity Dataset: STS-Benchmark
def load_sts_dataset(split="validation", limit=500):
    # STS-B gives sentence pairs and a score (0-5)
    # Example data:
    # {'sentence1': 'A man is dancing.', 'sentence2': 'A man is wearing a hat.', 'label': 5.0}
    # Scores range from 0.0 to 5.0
    dataset = load_dataset("glue", "stsb", split=split)
    dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset

# --- Helper functions to prepare data ---

def get_ner_texts(ner_ds):
    # NER data comes as a list of tokens: ['I', 'am', 'happy']
    # but spaCy expects a string: "I am happy"
    texts = []
    for item in ner_ds:
        texts.append(" ".join(item["tokens"]))
    return texts

def get_cls_texts(cls_ds):
    # get the text column
    texts = []
    for item in cls_ds:
        texts.append(item["text"])
    return texts

def get_sts_pairs(sts_ds):
    # return lists for sentence1, sentence2, and the gold scores
    s1_list = []
    s2_list = []
    scores = []
    
    for item in sts_ds:
        s1_list.append(item["sentence1"])
        s2_list.append(item["sentence2"])
        scores.append(item["label"])
        
    return s1_list, s2_list, scores