import re


def clean_cell_text(cell):
    if cell is None:
        return ""
    text = str(cell).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_text_for_match(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9%.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def is_year_or_period_header(text):
    text = normalize_text_for_match(text)
    if not text:
        return False
    return bool(
        re.search(r"\b(19|20)\d{2}\b", text) or
        re.search(r"\b\d{4}\s*%\b", text) or
        re.search(r"\b(year|period|march|april|may|june|july|august|september|october|november|december)\b", text)
    )


def choose_header_row(rows, max_scan=6):
    scan_rows = rows[:max_scan]
    best_idx = None
    best_score = -1

    for i, row in enumerate(scan_rows):
        non_empty = [c for c in row if c.strip()]
        if len(non_empty) < 2:
            continue

        score = 0
        for cell in row[1:]:
            cell = cell.strip()
            if not cell:
                continue
            score += 1
            if is_year_or_period_header(cell):
                score += 2
            if "%" in cell:
                score += 1

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        for i, row in enumerate(scan_rows):
            if sum(1 for c in row if c.strip()) >= 2:
                return i
        return None

    return best_idx


def is_section_row(row):
    non_empty = [(idx, c.strip()) for idx, c in enumerate(row) if c.strip()]
    if len(non_empty) == 1:
        return True

    if len(non_empty) == 2:
        first_text = non_empty[0][1]
        second_text = non_empty[1][1]
        if len(first_text) > 12 and len(second_text) <= 2:
            return True

    return False


def first_non_empty_cell(row):
    for cell in row:
        if str(cell).strip():
            return str(cell).strip()
    return None


def extract_table_chunks(table_obj, doc_id):
    chunks = []
    raw_rows = table_obj.get("table", []) if table_obj else []

    if not raw_rows:
        return chunks

    rows = [[clean_cell_text(cell) for cell in row] for row in raw_rows]
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return chunks

    header_idx = choose_header_row(rows)
    if header_idx is None:
        return chunks

    header = rows[header_idx]
    data_rows = rows[header_idx + 1:]

    normalized_header = []
    for j, h in enumerate(header):
        h = h.strip()
        normalized_header.append(h if h else f"column_{j}")

    current_section = None
    section_counter = 0

    for row_idx, row in enumerate(data_rows, start=header_idx + 1):
        if not any(cell.strip() for cell in row):
            continue

        if is_section_row(row):
            current_section = first_non_empty_cell(row)
            if current_section:
                section_counter += 1
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_type": "table_section",
                    "chunk_text": f"Section: {current_section}",
                    "section_label": current_section,
                    "row_label": None,
                    "column_header": None,
                    "value": None,
                })
            continue

        row_label = row[0].strip() if len(row) > 0 and row[0].strip() else first_non_empty_cell(row)
        if not row_label:
            continue

        row_cells = []

        for col_idx in range(1, len(row)):
            cell_value = row[col_idx].strip()
            if not cell_value:
                continue

            col_header = normalized_header[col_idx] if col_idx < len(normalized_header) else f"column_{col_idx}"

            parts = []
            if current_section:
                parts.append(f"Section: {current_section}")
            parts.append(f"Row: {row_label}")
            parts.append(f"Column: {col_header}")
            parts.append(f"Value: {cell_value}")

            cell_chunk = " | ".join(parts)
            row_cells.append(cell_chunk)

            chunks.append({
                "doc_id": doc_id,
                "chunk_type": "table_cell",
                "chunk_text": cell_chunk,
                "section_label": current_section,
                "row_label": row_label,
                "column_header": col_header,
                "value": cell_value,
            })

        if row_cells:
            row_prefix = []
            if current_section:
                row_prefix.append(f"Section: {current_section}")
            row_prefix.append(f"Row: {row_label}")
            row_chunk = " | ".join(row_prefix) + " | " + " ; ".join(
                [f"{c.split('|')[-2].strip()} | {c.split('|')[-1].strip()}" for c in row_cells]
            )

            chunks.append({
                "doc_id": doc_id,
                "chunk_type": "table_row",
                "chunk_text": row_chunk,
                "section_label": current_section,
                "row_label": row_label,
                "column_header": None,
                "value": None,
            })

    return chunks
