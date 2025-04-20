import pytesseract
from pdf2image import convert_from_path
import json
import os
from labeling import get_label

def process_pdf_ocr(pdf_path, output_json_path, poppler_path="/opt/homebrew/bin"):
    pages = convert_from_path(pdf_path, poppler_path=poppler_path)
    all_tokens, all_bboxes, all_labels = [], [], []

    for i, page in enumerate(pages):
        ocr_data = pytesseract.image_to_data(page, output_type=pytesseract.Output.DICT, config='--psm 6')
        text_lines = pytesseract.image_to_string(page).splitlines()

        n_boxes = len(ocr_data['text'])
        tokens, bboxes, labels = [], [], []

        prev_token = None
        for j in range(n_boxes):
            word = ocr_data['text'][j].strip()
            if word == "":
                continue

            x, y, w, h = ocr_data['left'][j], ocr_data['top'][j], ocr_data['width'][j], ocr_data['height'][j]
            bbox = [x, y, x + w, y + h]

            line_before, line_after = None, None
            for k, line in enumerate(text_lines):
                if word in line.split():
                    if k > 0:
                        line_before = text_lines[k - 1]
                    if k < len(text_lines) - 1:
                        line_after = text_lines[k + 1]
                    break

            label = get_label(word, bbox, prev_token=prev_token, line_before=line_before, line_after=line_after)

            tokens.append(word)
            bboxes.append(bbox)
            labels.append(label)
            prev_token = word

        all_tokens.extend(tokens)
        all_bboxes.extend(bboxes)
        all_labels.extend(labels)

    layoutlm_data = {
        "id": os.path.basename(pdf_path).replace(".pdf", ""),
        "tokens": all_tokens,
        "bboxes": all_bboxes,
        "labels": all_labels
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(layoutlm_data, f, indent=2)

    print(f"✅ JSON başarıyla oluşturuldu: {output_json_path}")
