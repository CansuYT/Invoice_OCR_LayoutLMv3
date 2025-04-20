from transformers import AutoTokenizer, LayoutLMv3ForTokenClassification, TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
from ocr_ner.create_dataset import load_dataset_from_json
from ocr_ner.ocr_process import process_pdf_ocr

# Tokenizer tanımla
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base")

# Pdfleri ocr ve ner işle ve json olarak kaydet
pdf_path = "/invoices"
output_dir = "/ocr_ner/json_outputs"  
for pdf_file in pdf_path:
    output_json = f"{output_dir}/{pdf_file.replace('.pdf', '.json')}"
    process_pdf_ocr(pdf_file, output_json)

# Modele uygun veri setini oluştur ve böl
dataset = load_dataset_from_json(output_dir)
dataset = dataset.train_test_split(test_size=0.1)

# Labellerin model için etiketlenmesi
label_list = sorted(set(['B-INVOICE_DATE_TAG','B-INVOICE_NUMBER_TAG','B-PO', 'B-PO_TAG', 'B-PRICE', 'B-PRICE_TAG','B-QUANTITY','B-QUANTITY_TAG',
                         'B-TOTAL','B-TOTAL_TAG','O']))
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

def tokenize_and_align_labels(example):
    if isinstance(example["labels"][0], list):
        example["labels"] = example["labels"][0]

    if len(example["labels"]) != len(example["tokens"]):
        return {}

    encoding = tokenizer(
        example["tokens"],
        boxes=example["bboxes"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

    word_ids = encoding.word_ids()
    labels = []
    for word_idx in word_ids:
        if word_idx is None or word_idx >= len(example["labels"]):
            labels.append(-100)
        else:
            labels.append(label2id[example["labels"][word_idx]])

    encoding["labels"] = labels
    return encoding

# Dataset tokenize
train_dataset = dataset["train"].map(
    tokenize_and_align_labels,
    batched=False,
    remove_columns=["id", "tokens", "bboxes"]
)
eval_dataset = dataset["test"].map(
    tokenize_and_align_labels,
    batched=False,
    remove_columns=["id", "tokens", "bboxes"]
)

# Performans metriklerinin hesaplanması için fonkisyon
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=2)

    true_labels, true_predictions = [], []

    for pred, label in zip(predictions, labels):
        for p_i, l_i in zip(pred, label):
            if l_i != -100:
                true_labels.append(id2label[l_i])
                true_predictions.append(id2label[p_i])

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average='micro'
    )
    acc = accuracy_score(true_labels, true_predictions)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Model tanımı
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./layoutlm_model",
    per_device_train_batch_size=1,
    num_train_epochs=5,
    evaluation_strategy="steps",
    learning_rate=1e-5,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    remove_unused_columns=False,
    metric_for_best_model="f1",
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics
)

# Eğitim başlat
trainer.train()
metrics = trainer.evaluate()

# Model kaydet
trainer.save_model("./layoutlmv3_invoice_ocr")
tokenizer.save_pretrained("./layoutlmv3_invoice_ocr")
