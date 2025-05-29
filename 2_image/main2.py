import os
import re
import time
import torch
import requests
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

OCR_POST_URL = "https://04a2-120-188-75-9.ngrok-free.app/documents/process"
OCR_GET_URL_BASE = "https://04a2-120-188-75-9.ngrok-free.app/documents/retrieve"

def load_internvl_model():
    path = 'OpenGVLab/InternVL2_5-1B'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        use_flash_attn=True if torch.cuda.is_available() else False,
        trust_remote_code=True
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer, device

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

def load_image(image_file, input_size=448):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16 if torch.cuda.is_available() else torch.float32).to(device)
    return pixel_values

def extract_images_and_context(markdown_path):
    with open(markdown_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    image_data = []
    for i, line in enumerate(lines):
        match = re.search(r'!\[.*?\]\((.*?)\)', line)
        if match:
            img_path = match.group(1)
            context_before = " ".join(lines[max(0, i-2):i]).strip()
            context_after = " ".join(lines[i+1:min(len(lines), i+3)]).strip()
            image_data.append((img_path, context_before, context_after))
    return image_data, lines

def generate_caption(model, tokenizer, image_path, context_before, context_after):
    if not os.path.exists(image_path):
        print(f"Warning: Image not found - {image_path}")
        return "[Image description unavailable]"
    pixel_values = load_image(image_path)
    prompt = f"<image>\nContext: {context_before} ... {context_after}. Please describe the image shortly."
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return response

def perform_ocr(image_path, timeout=60, interval=3):
    try:
        print(f"[OCR] Uploading image: {image_path}")
        with open(image_path, 'rb') as f:
            files = {
                'file': ('image.png', f, 'image/png')  # ✅ Explicit filename and MIME type
            }
            response = requests.post(OCR_POST_URL, files=files)

        response.raise_for_status()
        document_id = response.json()["data"]["documentId"]
        print(f"[OCR] Document ID received: {document_id}")

        # Polling loop
        elapsed = 0
        while elapsed < timeout:
            time.sleep(interval)
            elapsed += interval

            retrieve_url = f"{OCR_GET_URL_BASE}/{document_id}"
            retrieve_response = requests.get(retrieve_url)

            if retrieve_response.status_code != 200:
                print(f"[OCR] Poll failed: {retrieve_response.status_code}")
                continue

            result = retrieve_response.json()
            status = result["data"].get("status", "")
            print(f"[OCR] Status: {status}")

            if status == "COMPLETED":
                extraction = result["data"].get("extractionResult", {})
                ocr_text = extraction.get("text", "[OCR text unavailable]")
                ocr_summary = extraction.get("summary", "[OCR summary unavailable]")
                return ocr_text, ocr_summary

        return "[OCR timeout]", "[OCR summary timeout]"

    except Exception as e:
        print(f"[OCR] Error processing {image_path}: {e}")
        return "[OCR error]", "[OCR error]"


def update_markdown(markdown_path, image_data, lines, output_folder):
    new_lines = []
    for line in lines:
        new_lines.append(line)
        match = re.search(r'!\[.*?\]\((.*?)\)', line)
        if match:
            img_path = match.group(1)
            caption, ocr_text, ocr_summary = next(
                ((desc, text, summary) for img, _, _, desc, text, summary in image_data if img == img_path),
                ("[Image description unavailable]", "[OCR text unavailable]", "[OCR summary unavailable]")
            )
            new_lines.append(f"\n*Image Description:* {caption}\n")
            new_lines.append(f"*OCR Text:* {ocr_text}\n")
            new_lines.append(f"*OCR Summary:* {ocr_summary}\n")

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(markdown_path))
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def process_markdown_files(markdown_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model, tokenizer, device = load_internvl_model()

    for file in os.listdir(markdown_folder):
        if file.endswith(".md"):
            markdown_path = os.path.join(markdown_folder, file)
            filename_without_ext = os.path.splitext(file)[0]
            image_folder = os.path.join(markdown_folder, f"{filename_without_ext}_artifacts")

            if not os.path.exists(image_folder):
                print(f"Warning: Image folder '{image_folder}' not found for '{file}'")
                continue

            image_data, lines = extract_images_and_context(markdown_path)
            enriched_data = []
            for img_path, context_before, context_after in image_data:
                full_image_path = os.path.join(image_folder, img_path)
                caption = generate_caption(model, tokenizer, full_image_path, context_before, context_after)
                ocr_text, ocr_summary = perform_ocr(full_image_path)
                enriched_data.append((img_path, context_before, context_after, caption, ocr_text, ocr_summary))

            update_markdown(markdown_path, enriched_data, lines, output_folder)
            print(f"Processed: {file}")

if __name__ == "__main__":
    markdown_folder = r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\2_image\\input_md"
    output_folder = r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\2_image\\output_md"
    process_markdown_files(markdown_folder, output_folder)
