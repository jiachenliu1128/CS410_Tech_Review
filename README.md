# CS410_Tech_Review

Tech Review Project Repository of CS 410 Text Information Systems at University of Illinois Urbana-Champaign

Author: Jiachen Liang (liang88), Jiachen Liu (jl315)




# Overview
This repository contains code and documentation for evaluating the performance (performance, runtime, memory usage, model size) of different spaCy models on various NLP tasks, including Named Entity Recognition (NER), Text Classification, and Semantic Similarity.




# Models
You need to choose one of the following spaCy English models for `experiments.py`
- Standard Models:
    - Small: `en_core_web_sm`
    - Medium: `en_core_web_md`
    - Large: `en_core_web_lg`
- Transformer Model: `transformer`





# Mahcine Specifications
- CPU: AMD Ryzen™ 7 260
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- RAM: 16 GB
- OS: Windows 11
- Python version: 3.12.12






# Tasks
## Name Entity Recognition (NER)
- Identify named entities (people, organizations, locations, miscellaneous) in text and classify them with correct span boundaries.
- Dataset: CoNLL-2003 NER dataset (BIO-tagged PER, ORG, LOC, MISC), 10000 samples are used.
- CoNLL uses PER/ORG/LOC/MISC, but spaCy models are trained on OntoNotes using detailed labels with much more labels. This label mismatch lowers all metrics even though. But we are only comparing relative performance among different models, so it is acceptable.
- CoNLL provides token-level BIO tags, but spaCy requires character-span entities, so conversion is done in the code.
- Evaluation metrics: Precision, Recall, F1-score.

## Similarity
- Measure how similar two sentences are based on their embeddings (semantic textual similarity).
- Dataset: STS-B (Semantic Textual Similarity Benchmark) with about 5700 samples.
- Labeled similarity scores (0–5) and cosine similarity (–1 to 1) have different numeric scales → we evaluate using Pearson correlation, which is scale-invariant.
- For en_core_web_sm  has no word vectors, so the result of the Doc.similarity method will be based on the tagger, parser and NER.
- For transformer model, we extract the last layer hidden states as the embeddings and use cosine similarity to ensure consistency.
- Evaluation metric: Pearson correlation coefficient.

## Classification
- Classify news articles into one of categories that we defined.
- Dataset: AG News classification dataset with categories ['World', 'Sports', 'Business', 'Sci/Tech'], 20000 samples are used for training and 7600 samples are used for validation.
- spaCy’s pretrained English pipelines do not include a classifier, so we must manually add a TextCategorizer for each model.
    - Base modes can be trained on CPU in the `experiment.py` script, but transformer model training requires GPU with CUDA support.   
    - Instruction for training transformer classifier is provided at the end of this README.
- To play a fair game, we are using CPU for all models (including transformer) during evaluation.
- Evaluation metrics: Precision, Recall, F1-score.





# Transformer Classifier Training (before running `experiments.py`):
- Step 1: run `python cls_to_spacy.py` to convert the AG News dataset to spaCy format.
- Step 2: run the following commands to create config and train the transformer text classifier:
```python
python -m spacy init config ./configs/transformer.cfg \
    --lang en \
    --pipeline transformer,textcat \
    --optimize accuracy \
    --gpu
```
- Step 3: run the following command to train the transformer text classifier:
```python
python -m spacy train ./configs/transformer.cfg --gpu-id 0 -o models/transformer_textcat
```

    