from datasets import Dataset
import json

def load_dataset_from_json(json_paths):
    ids, tokens, bboxes, labels = [], [], [], []

    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            ids.append(data["id"])
            tokens.append(data["tokens"])
            bboxes.append(data["bboxes"])
            labels.append(data["labels"])

    dataset = Dataset.from_dict({
        "id": ids,
        "tokens": tokens,
        "bboxes": bboxes,
        "labels": labels
    })

    return dataset

def get_label_maps(label_list):
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label
