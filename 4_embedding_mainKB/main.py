import os
import re
import json
import uuid
import logging
import traceback
from pathlib import Path
from dotenv import load_dotenv

import torch
import boto3
import vecs
from botocore.client import Config
from supabase import create_client, Client
from transformers import AutoTokenizer, AutoModel

# ========== SETUP ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
load_dotenv()

# ========== ENV VARS ==========
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_STORAGE_ENDPOINT = os.getenv("SUPABASE_STORAGE_ENDPOINT")
SUPABASE_ACCESS_KEY_ID = os.getenv("SUPABASE_ACCESS_KEY_ID")
SUPABASE_SECRET_ACCESS_KEY = os.getenv("SUPABASE_SECRET_ACCESS_KEY")
SUPABASE_REGION = os.getenv("SUPABASE_REGION", "ap-southeast-1")
SUPABASE_BUCKET = "kb.files"
DB_CONNECTION = os.getenv("DB_CONNECTION")

# ========== SUPABASE CLIENTS ==========
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
s3 = boto3.client(
    "s3",
    aws_access_key_id=SUPABASE_ACCESS_KEY_ID,
    aws_secret_access_key=SUPABASE_SECRET_ACCESS_KEY,
    endpoint_url=SUPABASE_STORAGE_ENDPOINT,
    region_name=SUPABASE_REGION,
    config=Config(signature_version="s3v4"),
)

vx = vecs.create_client(DB_CONNECTION)
vec_text = vx.get_or_create_collection(name="kb.text", dimension=768)
vec_table = vx.get_or_create_collection(name="kb.table", dimension=768)

# ========== EMBEDDING MODEL ==========
logger.info("🔍 Loading embedding model...")
tokenizer_embed = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
model_embed = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# ========== TOC HELPER ==========
def slugify(text: str) -> str:
    """Generate markdown-compatible anchor from a heading"""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s]+", "-", text.strip())

def format_toc_items(toc_items: list) -> str:
    lines = ["## Table of Contents\n"]
    for item in toc_items:
        number = item.get("number", "-")
        title = item.get("title", "").strip()
        full_title = f"{number}. {title}"
        anchor = slugify(full_title)
        lines.append(f"- [{full_title}](#{anchor})")
    return "\n".join(lines)

# ========== CORE FUNCTIONS ==========

def get_embedding(text):
    inputs = tokenizer_embed(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model_embed.device)
    with torch.no_grad():
        outputs = model_embed(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()

def generate_table_description(table_data):
    headers = table_data["headers"]
    rows = table_data["rows"]
    return " | ".join([", ".join([f"{headers[i]}: {row[i]}" for i in range(len(headers))]) for row in rows])

def convert_table_to_text(table_data, metadata):
    headers = ", ".join(table_data["headers"])
    rows = [" | ".join(row) for row in table_data["rows"]]
    table_title = metadata.get("table_title", "Unknown Table")
    table_text = f"{table_title}\nHeaders: {headers}\nRows:\n" + "\n".join(rows)
    return table_text, generate_table_description(table_data)

def upload_images_to_bucket(chunks, artifacts_folder: Path, supabase_url, bucket_name="kb.files"):
    updated_chunks = []
    for chunk in chunks:
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown_source").replace(".md", "").replace(".pdf", "")
        image_matches = re.findall(r'!\[.*?\]\((.*?)\)', content)

        for img_rel_path in image_matches:
            if img_rel_path.startswith("http"):
                continue

            full_img_path = artifacts_folder / Path(img_rel_path).name

            if full_img_path.exists():
                storage_path = f"{source}/{full_img_path.name}"
                try:
                    with open(full_img_path, "rb") as f:
                        s3.upload_fileobj(f, SUPABASE_BUCKET, storage_path, ExtraArgs={"ContentType": "image/png"})

                    public_url = f"{supabase_url}/storage/v1/object/public/{bucket_name}/{storage_path}"
                    content = content.replace(img_rel_path, public_url)
                    logger.info(f"📤 Uploaded: {full_img_path.name} → {public_url}")
                except Exception:
                    logger.error(f"❌ Failed to upload image: {full_img_path}")
                    traceback.print_exc()
            else:
                logger.warning(f"⚠️ Image not found for upload: {full_img_path}")

        chunk["content"] = content
        updated_chunks.append(chunk)

    return updated_chunks

def store_chunks_in_supabase(chunks):
    doc_rows, tab_rows, text_records, table_records = [], [], [], []
    toc_items = []
    toc_chunk_idx = None
    toc_chunk_metadata = None
    toc_map = {}

    # Extract toc_items and remember index and metadata of the ToC chunk
    for i, chunk in enumerate(chunks):
        if "toc_items" in chunk:
            toc_items = chunk["toc_items"]
            toc_chunk_idx = i
            toc_chunk_metadata = chunk.get("metadata", {})
            break

    # Normalize and match ToC entry title with metadata section title
    def normalize(text):
        text = re.sub(r'^[\d\.]+\s*', '', text)  # remove leading numbers like "4.1.1"
        text = re.sub(r'\s+\d+$', '', text)        # remove trailing numbers like "Description 5"
        return re.sub(r"[^\w]+", " ", text).strip().lower()

    def match_chunk_to_toc(toc_items, section_text):
        norm_section = normalize(section_text)
        for item in toc_items:
            toc_norm = normalize(item["title"])
            if toc_norm == norm_section:
                return item["title"]
        return None

    # Generate and assign UUIDs before processing
    for chunk in chunks:
        chunk["chunk_id"] = str(uuid.uuid4())

    # Match chunks to ToC entries and track positions and IDs
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        section = metadata.get("section", "")
        position = metadata.get("position")
        chunk_id = chunk["chunk_id"]
        matched_title = match_chunk_to_toc(toc_items, section)

        if matched_title and position is not None:
            if matched_title not in toc_map:
                toc_map[matched_title] = {"positions": [], "chunk_ids": []}
            toc_map[matched_title]["positions"].append(position)
            toc_map[matched_title]["chunk_ids"].append(f'"{chunk_id}"')

    # Build formatted ToC content with position info and chunk IDs
    if toc_chunk_idx is not None:
        lines = ["## Table of Contents"]
        for item in toc_items:
            number = item.get("number", "-")
            title = item.get("title", "").strip()
            entry = toc_map.get(title)
            if entry:
                start_pos = min(entry["positions"])
                end_pos = max(entry["positions"])
                ids = ", ".join(entry["chunk_ids"])
                lines.append(f"- {number}. {title}: (\"section\": \"{title}\", \"position\": {start_pos}-{end_pos}, \"chunk_ids\": [{ids}])")
            else:
                lines.append(f"- {number}. {title}: (\"section\": \"{title}\", \"position\": none)")

        # Update content of ToC chunk
        chunks[toc_chunk_idx]["content"] = "\n".join(lines)

    # Upload chunks to Supabase
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        metadata = chunk.get("metadata", {})

        if "content" in chunk:
            content = chunk["content"]
            if content.strip():
                embedding = get_embedding(content)
                doc_rows.append({"chunk_id": chunk_id, "content": content, "metadata": metadata})
                text_records.append((chunk_id, embedding, metadata))

        if "table" in chunk:
            table_text, description = convert_table_to_text(chunk["table"], metadata)
            embedding = get_embedding(table_text)
            tab_rows.append({"chunk_id": chunk_id, "description": description, "metadata": metadata})
            table_records.append((chunk_id, embedding, metadata))

    if doc_rows:
        supabase.table("kb.text").insert(doc_rows).execute()
        vec_text.upsert(text_records)
        logger.info(f"✅ Uploaded {len(doc_rows)} text chunks to Supabase")

    if tab_rows:
        supabase.table("kb.table").insert(tab_rows).execute()
        vec_table.upsert(table_records)
        logger.info(f"✅ Uploaded {len(tab_rows)} table chunks to Supabase")

def upload_original_pdf(pdf_path: Path, supabase_url, bucket_name="kb.files"):
    if not pdf_path.exists():
        logger.error(f"❌ PDF file not found: {pdf_path}")
        return None, None

    file_name = pdf_path.name
    storage_dir = file_name.replace(".pdf", "")
    storage_path = f"{storage_dir}/{file_name}"

    try:
        with open(pdf_path, "rb") as f:
            s3.upload_fileobj(f, bucket_name, storage_path, ExtraArgs={"ContentType": "application/pdf"})

        public_url = f"{supabase_url}/storage/v1/object/public/{bucket_name}/{storage_path}"
        logger.info(f"📄 Uploaded original PDF: {file_name} → {public_url}")

        # Insert file metadata into public.kb.files
        supabase.table("kb.files").insert({
            "file_name": file_name,
            "file_path": public_url
        }).execute()

        return file_name, public_url
    except Exception:
        logger.error(f"❌ Failed to upload original PDF: {file_name}")
        traceback.print_exc()
        return None, None



# ========== MAIN RUNNER ==========
def process_json_chunks(json_path: Path, artifacts_folder: Path, original_pdf_path: Path):
    file_name, public_url = upload_original_pdf(original_pdf_path, SUPABASE_URL)
    if not file_name or not public_url:
        logger.warning("⚠️ Skipping chunk processing due to PDF upload failure.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"📄 Processing file: {json_path.name} with {len(chunks)} chunks")
    updated_chunks = upload_images_to_bucket(chunks, artifacts_folder, SUPABASE_URL)
    store_chunks_in_supabase(updated_chunks)

# ========== ENTRY POINT ==========
if __name__ == "__main__":
    # ✅ Define your inputs here
    JSON_FILE = Path(r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\4_embedding_store_copy\\input_json\\testing1.json")
    ARTIFACTS_FOLDER = Path(r"C:\\Users\\LENOVO\Desktop\\DOCKER-TRY\\1_parsing\\output_md\\testing1_artifacts")
    ORIGINAL_PDF = Path(r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\1_parsing\\input_pdfs\\testing1.pdf")

    process_json_chunks(JSON_FILE, ARTIFACTS_FOLDER, ORIGINAL_PDF)
