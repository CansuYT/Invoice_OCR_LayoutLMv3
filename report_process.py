import torch.nn.functional as F
from collections import defaultdict
import json


def extract_information(words, labels, confidence_scores=None):
    data = defaultdict(list)
    current_entity = None
    entity_tokens = []

    for word, label in zip(words, labels):
        if label.startswith("B-"):
            if current_entity:
                data[current_entity].append(" ".join(entity_tokens))
            current_entity = label[2:]
            entity_tokens = [word]
        elif label.startswith("I-") and current_entity:
            entity_tokens.append(word)
        else:
            if current_entity:
                data[current_entity].append(" ".join(entity_tokens))
                current_entity = None
                entity_tokens = []

    if current_entity and entity_tokens:
        data[current_entity].append(" ".join(entity_tokens))

    return data

def build_structured_output(words, labels, confidences=None):
    extracted_data = extract_information(words, labels, confidences)
    
    # PO Number kontrolü
    if "PO" in extracted_data:
        po_numbers = extracted_data["PO"]
        if po_numbers:  # PO listesi boş değilse
            print(f"✅ PO Numarası bulundu: {', '.join(po_numbers)}")
        else:
            print("❗ PO listesi boş!")
    else:
        print("❗ PO numarası bulunamadı!")


    # Tedarikçi kontrolü
    if "SUPPLIER" not in extracted_data:
        suplier = extracted_data["SUPPLIER"]
        if suplier:
            print(f"✅ Tedarikçi bilgisi bulundu: {', '.join(suplier)}")
        else:
            print("❗ Tedarikçi listesi boş!")
    else:
        print("❗ Tedarikçi bilgisi bulunamadı!")

    items = list(zip(
        extracted_data.get("QUANTITY", []),
        extracted_data.get("UNIT_PRICE", []),
        extracted_data.get("TOTAL_PRICE", [])
    ))

    consistent_count = 0
    total_count = len(items)
    for qty, unit, total in items:
        try:
            calc = round(float(qty) * float(unit), 2)
            real = round(float(total), 2)
            if abs(calc - real) <= 0.01:
                consistent_count += 1
        except:
            continue

    consistency_rate = (consistent_count / total_count * 100) if total_count > 0 else 0

    print(f"\nVeri Tutarlılığı Raporu:")
    print(f"- Toplam ürün kalemi: {total_count}")
    print(f"- Tutarlı satır sayısı: {consistent_count}")
    print(f"- Fiyat tutarlılığı: %{consistency_rate:.2f}")

    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        print(f"\nOrtalama model güven skoru: {avg_conf:.2f}")
        print("✅ Model doğruluğu yüksek!" if avg_conf > 0.90 else "⚠️ Model doğruluğu ortalama veya düşük!")

    structured_output = {
        "po_numbers": extracted_data.get("PO_NUMBER", []),
        "supplier": extracted_data.get("SUPPLIER", []),
        "items": [
            {
                "quantity": qty,
                "unit_price": unit,
                "total_price": total
            }
            for qty, unit, total in items
        ]
    }

    print("\nYapılandırılmış JSON Çıktısı:")
    print(json.dumps(structured_output, indent=4, ensure_ascii=False))
    return structured_output
