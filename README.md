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
- Dataset: CoNLL-2003 NER dataset (BIO-tagged PER, ORG, LOC, MISC), 5000 lines are used.
- CoNLL uses PER/ORG/LOC/MISC, but spaCy models are trained on OntoNotes using detailed labels with much more labels. This label mismatch lowers all metrics even though.
- CoNLL provides token-level BIO tags, but spaCy requires character-span entities, so conversion is done in the code.

## Classification
- Classify news articles into one of four categories: World, Sports, Business, SciTech.
- Dataset: AG News classification dataset with categories World, Sports, Business, SciTech, 5000 lines are used.
- spaCy’s pretrained English pipelines do not include a classifier, so we must manually add a TextCategorizer.

Must register labels manually (World, Sports, Business, SciTech).

Calling nlp.initialize() breaks pretrained components and triggers [E955] lexeme_norm errors — correct fix is to initialize only the textcat component.

Without training, the classifier predicts only one label → sklearn raises UndefinedMetricWarning.

Added a custom training loop (3 epochs, minibatching) to produce meaningful accuracy.

Transformer need to be trained separately

data split

# Similarity
What this task is:
Measure how similar two sentences are based on their embeddings (semantic textual similarity).

Dataset used:
STS-B (Semantic Textual Similarity Benchmark).

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

    