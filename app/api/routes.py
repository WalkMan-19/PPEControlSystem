import asyncio
import os
import tempfile
import cv2
from fastapi import APIRouter, UploadFile, File, Depends
import app.detection.singleton as detector_singleton
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_db
from fastapi.responses import FileResponse
import shutil

router = APIRouter()

def process_frame(detector, frame):
    """Синхронная функция для детекции на одном кадре (вызывается в потоке)."""
    return detector.detect(frame)

async def process_video_async(file_path: str, detector):
    """Асинхронная обёртка для обработки видеофайла без блокировки event loop."""
    loop = asyncio.get_running_loop()
    cap = cv2.VideoCapture(file_path)
    results = []
    frame_id = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detections = await loop.run_in_executor(None, process_frame, detector, frame)
            if detections:
                results.append({
                    "frame": frame_id,
                    "detections": detections
                })
            frame_id += 1
            await asyncio.sleep(0)
    finally:
        cap.release()
    return results

@router.post("/upload-video/")
async def upload_video(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """Принимает видеофайл, обрабатывает его моделью YOLOv8-P2 и возвращает найденные объекты."""
    detector = detector_singleton.detector_instance
    if detector is None:
        return {"error": "Модель детекции не загружена"}
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        results = await process_video_async(temp_path, detector)
        return {
            "filename": file.filename,
            "frames_processed": len(results),
            "detections": results
        }
    finally:
        os.remove(temp_path)
        os.rmdir(temp_dir)

@router.post("/upload-video-visualize/")
async def upload_video_visualize(file: UploadFile = File(...)):
    """Обрабатывает видео и возвращает файл с нарисованными рамками."""
    detector = detector_singleton.detector_instance
    if detector is None:
        return {"error": "Модель не загружена"}

    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    output_path = os.path.join(temp_dir, "output.avi")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cls_name = d["class_name"]
            conf = d["confidence"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        out.write(frame)

    cap.release()
    out.release()

    # очистка временной папки
    async def cleanup_async(path):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: shutil.rmtree(path, ignore_errors=True))

    return FileResponse(
        output_path,
        media_type="video/x-msvideo",
        filename="output.avi",
        background=lambda: cleanup_async(temp_dir)
    )