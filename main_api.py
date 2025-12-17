import os
import shutil
import uuid
import json
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from typing import Dict, Any

from config import settings
from processor import HeartRateProcessor

# --- Configuration ---

ALLOW_WEBCAM_FEATURE = True  # 设置为 False 可以禁用摄像头功能
# ---------------------


UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(title="rPPG Heart Rate API")

# --- Global instances & Data Structures ---

processor = HeartRateProcessor()
tasks: Dict[str, Dict] = {}
# -----------------------------------------


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def run_video_processing(task_id: str, video_path: str):
    """
    A background task to process the video, save results, and update task status.
    """
    try:
        result = processor.process(video_path)

        if "error" in result:
            tasks[task_id] = {"status": "error", "detail": result["error"]}
            return

        # 直接使用 processor 返回的结果，并进行必要的类型转换和清理
        result_data = result
        result_filepath = os.path.join(RESULTS_DIR, f"{task_id}.json")
        with open(result_filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)

        tasks[task_id] = {"status": "completed", "result_path": result_filepath}

    except Exception as e:
        tasks[task_id] = {"status": "error", "detail": f"处理过程中发生错误: {str(e)}"}
    finally:
        pass


@app.post("/predict/")
async def predict_heart_rate(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a video file, starts a background processing task, and returns a task ID.
    """
    task_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    video_filename = f"{task_id}{file_extension}"
    video_path = os.path.join(UPLOAD_DIR, video_filename)

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"无法保存上传文件: {str(e)}")
    finally:
        await file.close()

    tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(run_video_processing, task_id, video_path)

    return {"task_id": task_id}


@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """
    Polls for the status of a background task.
    """
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务ID未找到。")

    if task["status"] == "completed":
        result_path = task.get("result_path")
        if result_path and os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            return {"status": "completed", "result": result}
        else:
            return {"status": "error", "detail": "结果文件未找到。"}
            
    elif task["status"] == "error":
        return {"status": "error", "detail": task.get("detail", "未知错误。")}
        
    return {"status": "processing"}


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "allow_webcam": ALLOW_WEBCAM_FEATURE})


if __name__ == "__main__":
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
