"""
JFC Reflection Extractor
========================
Extracts the reflection text from all .docx and .pdf reflection files
across all week folders and saves them to a single JSON file.

Usage:
    python extract_reflections.py

Output:
    reflections.json  — saved in the same directory as this script

Requirements:
    pip install python-docx pypdf
"""

import os
import re
import json
import traceback
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────────────────────
# The script lives one directory up from the data folder.
BASE_DIR = Path(__file__).parent
OUTPUT_FILE = BASE_DIR / "reflections.json"

# Map physical folder names to week numbers
FOLDER_MAP = {
    "submissions": "Week 5",
    "submissions (2)": "Week 3",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _paragraph_in_table(para) -> bool:
    parent = para._p.getparent()
    while parent is not None:
        if parent.tag.endswith('}tc'):
            return True
        parent = parent.getparent()
    return False


def extract_text_from_docx(path: Path) -> str:
    """Extract all text from a .docx file, including table cells."""
    try:
        from docx import Document
        doc = Document(path)
        parts = []

        # Paragraphs outside of tables only; table text is handled separately.
        for para in doc.paragraphs:
            if para.text.strip() and not _paragraph_in_table(para):
                parts.append(para.text.strip())

        # Tables (reflection template often uses a table)
        for table in doc.tables:
            for row in table.rows:
                seen_cell_text = set()
                for cell in row.cells:
                    text = cell.text.strip()
                    if not text or text in seen_cell_text:
                        continue
                    seen_cell_text.add(text)
                    if parts and parts[-1] == text:
                        continue
                    parts.append(text)

        return "\n".join(parts)
    except Exception as e:
        return f"[DOCX ERROR: {e}]"


def extract_text_from_pdf(path: Path) -> str:
    """Extract all text from a .pdf file."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n".join(pages)
    except Exception as e:
        return f"[PDF ERROR: {e}]"


def extract_reflection_section(full_text: str) -> str:
    """
    Try to isolate just the reflection body from the full document text.
    The template has a 'Reflection' header followed by the student's writing.
    Falls back to returning the full text if no clear section is found.
    """
    # Common patterns that mark the start of the reflection
    start_patterns = [
        r"(?i)^reflection\s*$",
        r"(?i)weekly reflection\s*$",
        r"(?i)reflection\s*\n",
    ]
    # Patterns that mark the end of the reflection section
    end_patterns = [
        r"(?i)^weekly reflection\s*$",
        r"(?i)^supervisor\s*(comments|feedback|sign)",
        r"(?i)^please note",
        r"(?i)^references?\s*$",
    ]

    lines = full_text.splitlines()
    start_idx = None

    for i, line in enumerate(lines):
        for pat in start_patterns:
            if re.match(pat, line.strip()):
                start_idx = i + 1
                break
        if start_idx is not None:
            break

    if start_idx is None:
        return full_text.strip()

    # Find end marker after the start
    end_idx = len(lines)
    for i in range(start_idx, len(lines)):
        for pat in end_patterns:
            if re.match(pat, lines[i].strip()):
                end_idx = i
                break
        if end_idx != len(lines):
            break

    reflection = "\n".join(lines[start_idx:end_idx]).strip()
    if not reflection:
        reflection = full_text.strip()

    reflection = _deduplicate_blocks(reflection)
    return reflection


def _deduplicate_blocks(text: str) -> str:
    """
    Remove whole-block repetition caused by docx 3-column table extraction.
    Finds the first substantive sentence/line and checks if it reappears
    later in the text -- if so, truncate everything from that repeat onward.
    """
    lines_t = text.splitlines()

    # Find the first non-empty, non-header line that looks like real content
    # (longer than 40 chars to avoid matching short header lines like "Reflection")
    anchor = None
    anchor_idx = None
    for i, line in enumerate(lines_t):
        stripped = line.strip()
        if len(stripped) > 40:
            anchor = stripped
            anchor_idx = i
            break

    if anchor is None:
        return text

    # Look for a second occurrence of the anchor line after its first position
    for j in range(anchor_idx + 1, len(lines_t)):
        if lines_t[j].strip() == anchor:
            # Found repeat -- keep only text up to this point
            return "\n".join(lines_t[:j]).strip()

    return text


def parse_metadata_from_filename(filename: str) -> dict:
    """
    Extract student SID and name hint from the filename where possible.
    Example: abdullahmuhammad_359802_47211783_MuhammadAbdullah_Group_3_Week_1_Reflection.docx
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    sid = None
    for part in parts:
        if part.isdigit() and len(part) in (6, 7, 8):
            sid = part
            break
    return {"filename": filename, "sid": sid}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    results = []

    data_dir = BASE_DIR / "JFC_reflection part 2 2026 Jul"
    
    submission_dirs = [
        data_dir / "submissions",
        data_dir / "submissions (2)",
    ]

    # Keep only folders that actually exist
    submission_dirs = [d for d in submission_dirs if d.exists() and d.is_dir()]

    if not submission_dirs:
        print("No submission folders found.")
        print(f"Looked in: {data_dir}")
        return

    print(f"Found {len(submission_dirs)} submission folder(s)\n")

    for folder in submission_dirs:
        folder_name = folder.name
        week_label = FOLDER_MAP.get(folder_name, folder_name)

        doc_files = sorted(
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in (".docx", ".pdf")
        )

        print(f"{week_label} ({folder_name}): {len(doc_files)} file(s)")

        for file_path in doc_files:
            try:
                ext = file_path.suffix.lower()

                if ext == ".docx":
                    full_text = extract_text_from_docx(file_path)
                else:
                    full_text = extract_text_from_pdf(file_path)

                reflection_text = extract_reflection_section(full_text)
                meta = parse_metadata_from_filename(file_path.name)

                entry = {
                    "week": week_label,
                    "folder": folder_name,
                    "filename": file_path.name,
                    "sid": meta["sid"],
                    "file_type": ext.lstrip("."),
                    "reflection": reflection_text,
                    "full_text": full_text,
                }

                results.append(entry)

            except Exception:
                print(f"  ✗ Failed: {file_path.name}")
                traceback.print_exc()

    # Save output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Extracted {len(results)} reflection(s)")
    print(f"✓ Saved to: {OUTPUT_FILE}")

    # Summary
    weeks = {}
    for r in results:
        weeks.setdefault(r["week"], 0)
        weeks[r["week"]] += 1

    print("\nSummary:")
    for week, count in sorted(weeks.items()):
        print(f"  {week}: {count} reflections")
        
if __name__ == "__main__":
    main()