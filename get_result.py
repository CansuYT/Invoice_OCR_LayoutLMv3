import os
import json
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizerFast
import torch
import torch.nn.functional as F
from ocr_ner.ocr_process import process_pdf_ocr
from ocr_ner.create_dataset import load_dataset_from_json
from report_process import extract_information, build_structured_output

PDF_PATH = "invoices/example_invoice.pdf"
MODEL_PATH = "layoutlmv3_training/layoutlmv3_invoice_ocr"
OUTPUT_PATH = "."

# OCR ve Model için Veri Ön işleme
ocr_data = process_pdf_ocr(PDF_PATH)

ids, tokens, bboxes, labels = [], [], [], []
with open(ocr_data, "r", encoding="utf-8") as f:
    data = json.load(f)
    ids.append(data["id"])
    tokens.append(data["tokens"])
    bboxes.append(data["bboxes"])
    labels.append(data["labels"])
# Veri normalizayonu
def normalize_bbox(bbox, width, height):
    if isinstance(bbox[0], list):
        bbox = bbox[0]
    return [
        min(1000, max(0, int(1000 * bbox[0] / width))),
        min(1000, max(0, int(1000 * bbox[1] / height))),
        min(1000, max(0, int(1000 * bbox[2] / width))),
        min(1000, max(0, int(1000 * bbox[3] / height))),
    ]

def filter_and_normalize(data, image_width, image_height):
    filtered = [
        (word, box)
        for word, box in zip(data["tokens"], data["bboxes"])
        if word.strip() != "" and isinstance(box, list) and len(box) == 4
    ]
    tokens, bboxes = zip(*filtered)
    normalized_bboxes = [normalize_bbox(b, image_width, image_height) for b in bboxes]
    return tokens, normalized_bboxes

tokens, normalized_bboxes = filter_and_normalize(ocr_data, image_width=ocr_data["width"], image_height=ocr_data["height"])

# Tokenizer ve Model 
tokenizer = LayoutLMTokenizerFast.from_pretrained(MODEL_PATH)
model = LayoutLMForTokenClassification.from_pretrained(MODEL_PATH)
id2label = model.config.id2label

encoding = tokenizer(
    tokens,
    boxes=normalized_bboxes,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)

# Tahmin ve Güven Skorları 
def get_predictions_and_confidences(logits):
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = torch.max(probs, dim=-1)
    return predictions, confidences

model.eval()
with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits
    pred_ids, confidence_scores = get_predictions_and_confidences(logits)

pred_ids = pred_ids[0].cpu().tolist()
confidence_scores = confidence_scores[0].cpu().tolist()
predicted_labels = [id2label[pred_id] for pred_id in pred_ids]

# Raporlama ve JSON çıktısı
words = encoding.tokens()[0:len(predicted_labels)]
structured_output = build_structured_output(words, predicted_labels, confidence_scores)

# Kaydetme
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(structured_output, f, ensure_ascii=False, indent=4)

print("\nİşlem tamamlandı. Sonuçlar:", OUTPUT_PATH)
