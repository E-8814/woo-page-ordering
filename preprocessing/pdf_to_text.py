#!/usr/bin/env python3
"""
PDF → Text extractor (no embeddings)

PDF_TO_TEXT_MAX_WORKERS=63 python3 pdf_to_text.py

This script extracts text from each page of every PDF in `documents_to_embed/`.
It cleans the text and saves either one combined .txt per PDF and/or one .txt per page.

Dependencies: PyMuPDF (fitz)
    pip install pymupdf

Outputs (configurable):
- extracted_text/<PDF_NAME>.txt              (combined text)
- extracted_text_pages/<PDF_NAME>_page_X.txt (per-page text)
"""

import os
import re
import sys
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import threading
import time

import fitz  # PyMuPDF
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR not available. Install with: pip install pytesseract pillow")


# ==========================
# Configuration
# ==========================
DOCUMENTS_DIR = "documents"  # Folder that contains input PDFs

# Save a single combined .txt per PDF
SAVE_COMBINED_PER_PDF = False
COMBINED_OUT_DIR = "extracted_text"

# Also save one .txt per page
SAVE_PER_PAGE = True
PER_PAGE_OUT_DIR = "text_pages"

# Skip writing placeholder lines for pages where no text could be extracted
SKIP_EMPTY_PAGE_PLACEHOLDER = True

# OCR Configuration (when regular text extraction fails)
USE_OCR_FALLBACK = True
OCR_DPI = 800 # Higher = better quality, slower (300-600 recommended)
OCR_LANGUAGE = "nld"  # Change to your language code (e.g., "eng+deu" for English+German)

# Debug/Preview Configuration
PRINT_EXTRACTED_TEXT = False  # Printing is expensive for huge batches; disable by default
MAX_PRINT_LENGTH = 0      # Max characters to print per page (0 = no limit)

# Parallel Processing Configuration
USE_PARALLEL_PROCESSING = True  # Enable parallel processing

# Allow controlling worker count via env var; default to all cores
MAX_WORKERS = int(os.getenv("PDF_TO_TEXT_MAX_WORKERS", str(cpu_count())))

# Work distribution strategy:
# - 'file': parallelize by PDF file (original behavior)
# - 'page': parallelize by page across all PDFs (better for many-core machines)
# - 'auto': choose 'page' if fewer PDFs than workers, else 'file'
TASK_DISTRIBUTION = os.getenv("PDF_TO_TEXT_TASK_DISTRIBUTION", "auto").lower()

# For page-level execution, submit small batches to reduce overhead
PAGE_TASK_CHUNKSIZE = int(os.getenv("PDF_TO_TEXT_PAGE_CHUNKSIZE", "4"))

# Limit Tesseract's internal threading to avoid oversubscription when many workers run OCR
os.environ.setdefault("OMP_THREAD_LIMIT", os.getenv("TESSERACT_MAX_THREADS", "1"))


# ==========================
# Text extraction helpers
# ==========================
def clean_text(text: str) -> str:
    """Clean text by removing control/zero-width chars and collapsing whitespace."""
    # Replace control characters and zero-width spaces
    text = re.sub(r"[\x00-\x1F\x7F-\x9F\u200B-\u200D\uFEFF]", "", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove non-printable characters (keep spaces)
    text = "".join(c for c in text if c.isprintable() or c.isspace())
    return text


def process_page(page: fitz.Page) -> str:
    """Extract text from a single PDF page using multiple strategies with fallbacks."""
    try:
        text = ""

        # 1) Plain text (fastest)
        text_mode = page.get_text("text")
        if len(text_mode.strip()) > 0:
            text = text_mode

        # 2) HTML (strip tags) if text is still short
        if len(text.strip()) < 50:
            html_mode = page.get_text("html")
            if len(html_mode) > len(text_mode):
                html_mode = re.sub(r"<[^>]*>", " ", html_mode)
                text = html_mode

        # 3) JSON blocks if still short
        if len(text.strip()) < 50:
            try:
                json_mode = page.get_text("json")
                import json as json_lib
                json_data = json_lib.loads(json_mode)
                json_text = " ".join([blk.get("text", "") for blk in json_data.get("blocks", [])])
                if len(json_text) > len(text):
                    text = json_text
            except Exception:
                # JSON extraction failed; continue with what we have
                pass

        # 4) XHTML (strip tags) if still short
        if len(text.strip()) < 50:
            try:
                xhtml_mode = page.get_text("xhtml")
                xhtml_mode = re.sub(r"<[^>]*>", " ", xhtml_mode)
                if len(xhtml_mode) > len(text):
                    text = xhtml_mode
            except Exception:
                pass

        # 5) OCR fallback if still short and OCR is available
        if len(text.strip()) < 50 and USE_OCR_FALLBACK and OCR_AVAILABLE:
            try:
                # Render page as high-resolution image
                zoom = OCR_DPI / 72.0  # Convert DPI to zoom factor
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                
                # Run OCR
                ocr_text = pytesseract.image_to_string(img, lang=OCR_LANGUAGE)
                
                # Use OCR result if it's better
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    print(f"    Used OCR for better text extraction")
                
            except Exception as e:
                print(f"    OCR failed: {e}")

        return clean_text(text)
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""


def process_pdf_file(pdf_path: str) -> List[str]:
    """Extract cleaned text for each page in the given PDF."""
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        pages_text: List[str] = []

        print(f"Extracting text from {pdf_path} ({total_pages} pages)")

        for i in range(total_pages):
            try:
                text = process_page(doc[i])
                if text.strip():
                    pages_text.append(text)
                    
                    # Print extracted text if enabled
                    if PRINT_EXTRACTED_TEXT:
                        print(f"\n--- Page {i + 1} Text ---")
                        if MAX_PRINT_LENGTH > 0 and len(text) > MAX_PRINT_LENGTH:
                            print(text[:MAX_PRINT_LENGTH] + "... [truncated]")
                        else:
                            print(text)
                        print("--- End Page Text ---\n")
                else:
                    # Keep placeholder if we need to preserve page count
                    pages_text.append("No extractable text on this page.")
                    if PRINT_EXTRACTED_TEXT:
                        print(f"\n--- Page {i + 1}: No extractable text ---\n")
                        
                if (i + 1) % 10 == 0 or i == total_pages - 1:
                    print(f"  Processed {i + 1}/{total_pages} pages")
            except Exception as e:
                print(f"Error processing page {i + 1}: {e}")
                pages_text.append("No extractable text on this page.")

        doc.close()
        return pages_text
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return []


def extract_page_from_path(args) -> Tuple[str, int, str]:
    """Extract cleaned text from a single page given (pdf_path, page_index).

    Returns (pdf_file_name, page_index_zero_based, text).
    """
    pdf_path, page_index = args
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        text = process_page(page)
        doc.close()
        return os.path.basename(pdf_path), page_index, text
    except Exception as e:
        print(f"Error extracting page {page_index + 1} from {pdf_path}: {e}")
        return os.path.basename(pdf_path), page_index, ""


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def process_single_pdf(args) -> tuple:
    """Process a single PDF file - designed for parallel execution."""
    pdf_file, documents_dir = args
    pdf_path = os.path.join(documents_dir, pdf_file)
    
    print(f"🔄 Starting: {pdf_file}")
    start_time = time.time()
    
    try:
        pages_text = process_pdf_file(pdf_path)
        elapsed = time.time() - start_time
        
        if not pages_text:
            print(f"⚠️  Finished: {pdf_file} (no text, {elapsed:.1f}s)")
            return pdf_file, None, f"No text extracted from {pdf_file}"
        
        print(f"✅ Finished: {pdf_file} ({len(pages_text)} pages, {elapsed:.1f}s)")
        return pdf_file, pages_text, None
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Failed: {pdf_file} ({elapsed:.1f}s) - {e}")
        return pdf_file, None, f"Error processing {pdf_file}: {e}"


def write_outputs(pdf_file: str, pages_text: List[str]) -> None:
    """Write combined and/or per-page text files based on configuration."""
    base_name = os.path.splitext(pdf_file)[0]

    if SAVE_COMBINED_PER_PDF:
        ensure_dir(COMBINED_OUT_DIR)
        combined = []
        for t in pages_text:
            if SKIP_EMPTY_PAGE_PLACEHOLDER and t == "No extractable text on this page.":
                continue
            combined.append(t)
        combined_text = "\n\n".join(combined)
        out_path = os.path.join(COMBINED_OUT_DIR, f"{base_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(combined_text)

    if SAVE_PER_PAGE:
        ensure_dir(PER_PAGE_OUT_DIR)
        for idx, t in enumerate(pages_text, start=1):
            if SKIP_EMPTY_PAGE_PLACEHOLDER and t == "No extractable text on this page.":
                continue
            out_path = os.path.join(PER_PAGE_OUT_DIR, f"{base_name}_page_{idx}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(t)


def write_single_page_output(pdf_file: str, page_index_zero_based: int, text: str) -> None:
    """Write a single page's text to disk respecting configuration flags."""
    if not SAVE_PER_PAGE:
        return
    if SKIP_EMPTY_PAGE_PLACEHOLDER and not text.strip():
        return

    ensure_dir(PER_PAGE_OUT_DIR)
    base_name = os.path.splitext(pdf_file)[0]
    page_number = page_index_zero_based + 1
    out_path = os.path.join(PER_PAGE_OUT_DIR, f"{base_name}_page_{page_number}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text if text.strip() else "No extractable text on this page.")


def main() -> None:
    if not os.path.exists(DOCUMENTS_DIR):
        print(f"Directory not found: {DOCUMENTS_DIR}")
        return

    pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {DOCUMENTS_DIR}")
        return

    # Sort for deterministic output order
    pdf_files.sort()
    print(f"Found {len(pdf_files)} PDF files")
    
    if USE_PARALLEL_PROCESSING:
        # Decide strategy
        strategy = TASK_DISTRIBUTION if TASK_DISTRIBUTION in {"file", "page", "auto"} else "auto"
        if strategy == "auto":
            strategy = "page" if len(pdf_files) < MAX_WORKERS else "file"

        if strategy == "file" and len(pdf_files) > 1:
            print(f"🚀 Parallel (by file) with {MAX_WORKERS} workers | {len(pdf_files)} PDFs")

            pdf_args = [(pdf_file, DOCUMENTS_DIR) for pdf_file in pdf_files]
            completed_count = 0
            total_count = len(pdf_files)
            start_time = time.time()

            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_pdf = {executor.submit(process_single_pdf, args): args[0] for args in pdf_args}

                for future in as_completed(future_to_pdf):
                    pdf_file = future_to_pdf[future]
                    completed_count += 1
                    elapsed_total = time.time() - start_time

                    try:
                        pdf_name, pages_text, error = future.result()
                        if error:
                            continue
                        if pages_text:
                            write_outputs(pdf_name, pages_text)

                        remaining = total_count - completed_count
                        avg_time_per_pdf = elapsed_total / max(completed_count, 1)
                        eta_seconds = avg_time_per_pdf * remaining
                        eta_str = f"{eta_seconds/60:.1f}min" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                        print(
                            f"📈 Progress: {completed_count}/{total_count} ({completed_count/total_count*100:.1f}%) | "
                            f"Elapsed: {elapsed_total/60:.1f}min | ETA: {eta_str}"
                        )

                    except Exception as e:
                        print(f"❌ Unexpected error with {pdf_file}: {e}")

        else:
            # Page-level parallelism across all PDFs
            print(f"🚀 Parallel (by page) with {MAX_WORKERS} workers | preparing page tasks…")

            # Precompute page counts
            pdf_to_num_pages = {}
            total_pages = 0
            for pdf_file in pdf_files:
                pdf_path = os.path.join(DOCUMENTS_DIR, pdf_file)
                try:
                    with fitz.open(pdf_path) as doc:
                        n_pages = len(doc)
                    pdf_to_num_pages[pdf_file] = n_pages
                    total_pages += n_pages
                except Exception as e:
                    print(f"Error opening {pdf_file} to count pages: {e}")

            print(f"📊 Total pages to process: {total_pages}")
            processed_pages = 0
            start_time = time.time()

            def task_iter():
                for pdf_file, n_pages in pdf_to_num_pages.items():
                    pdf_path = os.path.join(DOCUMENTS_DIR, pdf_file)
                    for idx in range(n_pages):
                        yield (pdf_path, idx)

            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for pdf_name, page_index, text in executor.map(
                    extract_page_from_path, task_iter(), chunksize=PAGE_TASK_CHUNKSIZE
                ):
                    processed_pages += 1

                    if PRINT_EXTRACTED_TEXT and text:
                        print(f"\n--- {pdf_name} Page {page_index + 1} Text ---")
                        if MAX_PRINT_LENGTH > 0 and len(text) > MAX_PRINT_LENGTH:
                            print(text[:MAX_PRINT_LENGTH] + "... [truncated]")
                        else:
                            print(text)
                        print("--- End Page Text ---\n")

                    write_single_page_output(pdf_name, page_index, text)

                    if processed_pages % 25 == 0 or processed_pages == total_pages:
                        elapsed_total = time.time() - start_time
                        avg_time_per_page = elapsed_total / max(processed_pages, 1)
                        remaining = total_pages - processed_pages
                        eta_seconds = avg_time_per_page * remaining
                        eta_str = f"{eta_seconds/60:.1f}min" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
                        print(
                            f"📈 Progress: {processed_pages}/{total_pages} ({processed_pages/total_pages*100:.1f}%) | "
                            f"Elapsed: {elapsed_total/60:.1f}min | ETA: {eta_str}"
                        )
    else:
        # Sequential processing (original method)
        print("Using sequential processing")
        for pdf_file in pdf_files:
            pdf_path = os.path.join(DOCUMENTS_DIR, pdf_file)
            pages_text = process_pdf_file(pdf_path)
            if not pages_text:
                print(f"No text extracted from {pdf_file}, skipping...")
                continue
            write_outputs(pdf_file, pages_text)

    print("\nDone.")
    if SAVE_COMBINED_PER_PDF:
        print(f"Combined text saved under: {COMBINED_OUT_DIR}/")
    if SAVE_PER_PAGE:
        print(f"Per-page text saved under: {PER_PAGE_OUT_DIR}/")


if __name__ == "__main__":
    # For large PDFs, PyMuPDF can be recursive when parsing; increase recursion limit a bit
    sys.setrecursionlimit(10_000)
    main()

