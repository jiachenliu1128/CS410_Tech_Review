# CS410_Tech_Review

Tech Review Project Repository of CS 410 Text Information Systems at University of Illinois Urbana-Champaign

Author: Jiachen Liang (liang88), Jiachen Liu (jl315)

# Models
- Standard Models:
    - Small: `en_core_web_sm`
    - Medium: `en_core_web_md`
    - Large: `en_core_web_lg`
- Transformer Models:
    - RoBERTa-base: `transformer`

# Tasks
## Name Entity Recognition (NER)
- Identify named entities (people, organizations, locations, miscellaneous) in text and classify them with correct span boundaries.
- Dataset: CoNLL-2003 NER dataset (BIO-tagged PER, ORG, LOC, MISC), 10000 samples are used.
- CoNLL uses PER/ORG/LOC/MISC, but spaCy models are trained on OntoNotes using detailed labels with much more labels. This label mismatch lowers all metrics even though.
- CoNLL provides token-level BIO tags, but spaCy requires character-span entities, so conversion is done in the code.

## Classification
- Classify news articles into one of categories that we defined.
- Dataset: AG News classification dataset with categories ['World', 'Sports', 'Business', 'Sci/Tech'], 20000 samples are used for training and 7600 samples are used for validation.
- spaCy’s pretrained English pipelines do not include a classifier, so we must manually add a TextCategorizer for each model.
- Base modes can be trained on CPU in the `experiment.py` script, but transformer model training requires GPU with CUDA support.
- Instruction for training transformer classifier is provided at the end of this README.

## Similarity
What this task is:
Measure how similar two sentences are based on their embeddings (semantic textual similarity).

Dataset used:
STS-B (Semantic Textual Similarity Benchmark) with about 5700 samples.

Issues / Challenges:

Human similarity scores (0–5) and cosine similarity (–1 to 1) have different numeric scales → we evaluate using Pearson correlation, which is scale-invariant.

spaCy standard model uses static word embeddings, while spaCy transformer uses contextual embeddings, so performance differs significantly.

Need consistent vector extraction (doc.vector) across both models for fair comparison.

Must ensure models actually support vectors — older spaCy models or disabled components will cause empty vectors.

Extract last layer for transformer embeddings and use cosine similarity to ensure consistency.

# Notes when writing the report:
- Need to mention the configuration of our machine. In order to play a fair game, we are using CPU for all models (includeing transformer). GPU is used for transformer training.

# Transformer Classifier Training Commands:
```python
python -m spacy init config ./configs/transformer.cfg \
    --lang en \
    --pipeline transformer,textcat \
    --optimize accuracy \
    --gpu
```

```python
python -m spacy train ./configs/transformer.cfg --gpu-id 0 -o models/transformer_textcat
```

    