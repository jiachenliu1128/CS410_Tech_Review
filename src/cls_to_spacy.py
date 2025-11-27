import spacy
from spacy.tokens import DocBin
from dataloader import load_cls_dataset
from data_formatter import format_cls_dataset


def convert_split_to_spacy(split: str, output_path: str, limit: int = 50000):
    """
    Convert the AG News classification split into spaCy's .spacy format.

    Each Doc will have:
        doc.text = news text
        doc.cats = one-hot dict over label names, e.g.
                   {"World": 1.0, "Sports": 0.0, "Business": 0.0, "Sci/Tech": 0.0}
    """
    print(f"Loading AG News {split} split...")
    cls_ds = load_cls_dataset(split=split, limit=limit)

    # Get label names from the original HF dataset
    label_names = cls_ds.features["label"].names
    print("Label names:", label_names)

    print("Formatting dataset with your format_cls_dataset()...")
    formatted = format_cls_dataset(cls_ds)
    print(f"Number of examples in {split}: {len(formatted)}")

    # We only need a tokenizer here
    nlp = spacy.blank("en")
    doc_bin = DocBin(store_user_data=True)

    for ex in formatted:
        text = ex["text"]
        label = ex["label"] 
        doc = nlp.make_doc(text)

        # one-hot cats over all label names
        cats = {lbl: 0.0 for lbl in label_names}
        cats[label] = 1.0
        doc.cats = cats

        doc_bin.add(doc)

    output_path = spacy.util.ensure_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc_bin.to_disk(output_path)
    print(f"Saved {split} split to {output_path}")


if __name__ == "__main__":
    # Train split
    convert_split_to_spacy(
        split="train",
        output_path="data/cls_train.spacy",
        limit=20000,
    )

    # Use AG News test split as dev
    convert_split_to_spacy(
        split="test",
        output_path="data/cls_dev.spacy",
        limit=10000,
    )