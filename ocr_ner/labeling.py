import re

def get_label(token, bbox, prev_token=None, line_before=None, line_after=None):
    token_lower = token.lower()

    if token_lower in ["qty", "qty:", "quantity", "quantity:", "hours", "hours:"]:
        return "B-QUANTITY_TAG"
    elif token_lower in ["price", "price:", "unit price", "unit price:", "rate", "rate:"]:
        return "B-PRICE_TAG"
    elif token_lower in ["total", "total:", "amount", "amount:"]:
        return "B-TOTAL_TAG"
    elif token_lower in ["invoice", "invoice:", "invoice number", "invoice number:"]:
        return "B-INVOICE_NUMBER_TAG"
    elif token_lower in ["date", "date:", "invoice date", "invoice date:"]:
        return "B-INVOICE_DATE_TAG"
    elif token_lower in ["po", "po:", "po number", "po number:"]:
        return "B-PO_TAG"
    elif token_lower in ["supplier", "supplier:", "vendor", "vendor:"]:
        return "B-SUPPLIER_TAG"

    if re.match(r"^(po[-:\s]?)?\d{5,}$", token_lower):
        return "B-PO"

    if re.match(r"^\d+$", token) and prev_token and prev_token.lower() in ["qty", "quantity", "hours", "qty:", "quantity:", "hours:"]:
        return "B-QUANTITY"

    if re.match(r"^\$?[\d,]+(\.\d{2})?$", token) and prev_token and prev_token.lower() in ["price", "unit price", "rate", "price:", "unit price:", "rate:"]:
        return "B-PRICE"

    if re.match(r"^\$?[\d,]+(\.\d{2})?$", token) and prev_token and prev_token.lower() in ["total", "amount", "total:", "amount:"]:
        return "B-TOTAL"

    return "O"
