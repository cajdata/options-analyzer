# app/main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

from app.ocr import extract_text_from_image
from app.parse import parse_ocr_text
from app.analysis import analyze_option

app = FastAPI()

# Mount the "public" folder at the "/public" path
app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    """
    Serve the index.html at the root URL.
    """
    index_file_path = os.path.join("public", "index.html")
    return FileResponse(index_file_path)

@app.post("/analyze")
async def analyze_screenshot(file: UploadFile = File(...)):
    """
    Receives an uploaded screenshot, performs OCR,
    parses data, then analyzes the option’s investment potential.
    """
    # 1) Read the file into memory
    contents = await file.read()

    # 2) Convert bytes to text via OCR
    try:
        ocr_text = extract_text_from_image(contents)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to process image: {str(e)}"}
        )

    # 3) Parse the relevant fields
    parsed_data = parse_ocr_text(ocr_text)

    # 4) Analyze the data
    analysis_result = analyze_option(parsed_data)

    return {
        "parsed_data": parsed_data,
        "analysis_result": analysis_result,
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
