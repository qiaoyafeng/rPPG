
import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

from processor import HeartRateProcessor

# --- Configuration ---
ALLOW_WEBCAM_FEATURE = True # 设置为 False 可以禁用摄像头功能
# ---------------------

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="rPPG Heart Rate API")

# --- Global instances ---
# Load the model and processor once when the application starts
processor = HeartRateProcessor()
# ----------------------

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.post("/predict/")
async def predict_heart_rate(file: UploadFile = File(...)):
    """
    Accepts a video file, processes it, and returns the calculated heart rate.
    """
    temp_video_path = None
    try:
        # Save the uploaded video file temporarily
        temp_video_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Use the global processor instance
        heart_rate = processor.process(temp_video_path)

        if heart_rate == -1.0:
            raise HTTPException(status_code=400, detail="视频太短，无法处理。")
        if heart_rate == 0.0:
            raise HTTPException(status_code=400, detail="无法处理视频或未找到有效帧。")

        # Return the result
        return {"heart_rate": int(heart_rate), "units": "bpm"}

    except Exception as e:
        # Catch any other exceptions from the processing
        raise HTTPException(status_code=500, detail=f"处理过程中发生错误: {str(e)}")

    finally:
        # Clean up the uploaded file
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        # Close the file handle
        if file:
            await file.close()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "allow_webcam": ALLOW_WEBCAM_FEATURE})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
