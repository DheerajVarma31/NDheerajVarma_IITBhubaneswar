import os
import io
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
import google.generativeai as genai
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



os.environ["GEMINI_API_KEY"] = "AIzaSyCgJgKewsS_-q5vyRopR8yk1I_UMgWvskw"  
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")


 
app = FastAPI(title="Bill Extractor API using Gemini")



def extract_text(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)



def llm_parse(text: str):
    prompt = f"""
    Extract bill items from this bill text and return format:

    {{
      "items": [
        {{"name": "...", "qty": number, "price": number}}
      ],
      "total_items": number
    }}

    Text: {text}
    """

    result = model.generate_content(prompt)
    return result.text



@app.post("/extract-bill-data")
async def extract_bill_data(file: UploadFile = File(...)):
    """
    Upload a bill image and get structured bill data.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image (PNG, JPG, JPEG, etc.)."
        )

    try:
        file_bytes = await file.read()

        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Could not read image. Please upload a valid image file."
            )

        ocr_text = extract_text(image)

        llm_raw_output = llm_parse(ocr_text)

        try:
            extracted_data = json.loads(llm_raw_output)
        except Exception:
            extracted_data = llm_raw_output

        return JSONResponse(
            {
                "success": True,
                "message": "Bill processed successfully.",
                "file_name": file.filename,
                "ocr_text": ocr_text,
                "extracted_data": extracted_data
            }
        )

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process bill: {str(error)}"
        )


@app.get("/")
def home():
    return {
        "status": "Bill Extraction API is live ",
        "info": "POST an image to /extract-bill-data to extract bill items.",
        "docs": "/docs"
    }