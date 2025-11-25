# Helpers to convert conll datasets to spaCy format
def conll_to_spacy_example(ex, label_names):
    tokens = ex["tokens"]
    tag_ids = ex["ner_tags"]

    text = " ".join(tokens)
    entities = []

    char_pos = 0        
    current_start = None
    current_label = None

    for token, tag_id in zip(tokens, tag_ids):
        tag = label_names[tag_id] 
        
        # find token in the reconstructed text
        token_start = text.find(token, char_pos)
        token_end = token_start + len(token)

        if tag == "O":
            # if we were inside an entity, close it
            if current_start is not None:
                entities.append((current_start, prev_end, current_label))
                current_start = None
                current_label = None

        elif tag.startswith("B-"):
            # if we were inside another entity, close it
            if current_start is not None:
                entities.append((current_start, prev_end, current_label))

            current_start = token_start
            current_label = tag.split("-", 1)[1]  # "B-ORG" → "ORG"
            if current_label == "PER":
                current_label = "PERSON" # spaCy uses "PERSON" instead of "PER"

        elif tag.startswith("I-"):
            # continuation of the current entity; nothing to start,
            # we just extend by updating prev_end below
            pass

        # remember the last token's end; useful when we close
        prev_end = token_end
        # move char_pos forward so next search doesn't start at 0 again
        char_pos = token_end

    # if we ended still inside an entity, close it
    if current_start is not None:
        entities.append((current_start, prev_end, current_label))

    return {"text": text, "entities": entities}

def format_ner_dataset(ner_ds):
    # convert the conll dataset to spaCy format
    formatted_dataset = []
    for i, d in enumerate(ner_ds):
        formatted_dataset.append(conll_to_spacy_example(d, ner_ds.features["ner_tags"].feature.names))
    return formatted_dataset





# Helper to convert ag_news datasets to spaCy format
def format_cls_dataset(cls_ds):
    label_names = cls_ds.features["label"].names
    formatted_dataset = []
    for i, d in enumerate(cls_ds):
        formatted_dataset.append({
            "text": d["text"],
            "label": label_names[d["label"]]
        })
    return formatted_dataset





# Helper to convert sts-b datasets to spaCy format
def format_sts_dataset(sts_ds):
    return sts_ds