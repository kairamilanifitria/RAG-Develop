from fastapi import FastAPI
import os
import regex  # Import from main.py

app = FastAPI()

# Set your fixed input and output directories
INPUT_FOLDER = r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\3_chunking\\input_md"
OUTPUT_FOLDER = r"C:\\Users\\LENOVO\\Desktop\\DOCKER-TRY\\3_chunking\\output_json"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.get("/chunking")
def chunk_all_markdown_files():
    """Process all markdown files in input_md and save the results to output_json."""
    processed_files = []
    
    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.endswith(".md"):
            input_path = os.path.join(INPUT_FOLDER, file_name)
            # Let main.py use its own default output logic
            regex.process_markdown(input_path)
            processed_files.append(file_name)

    return {
        "message": "✅ Processing completed.",
        "processed_files": processed_files,
        "output_dir": OUTPUT_FOLDER
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8200)
