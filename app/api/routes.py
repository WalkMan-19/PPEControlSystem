import asyncio
import os
import shutil
import tempfile
import cv2
from fastapi import APIRouter, UploadFile, File, Depends
from fastapi.responses import FileResponse
from app.detection import singleton as detector_singleton
from app.api.dependencies import get_db
from app.detection.config import CLASS_COLORS, DEFAULT_COLOR
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()

def process_frame(detector, frame):
    """Синхронная функция для детекции на одном кадре."""
    return detector.detect(frame)


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
            cls_name = d["class_name"]
            x1, y1, x2, y2 = d["bbox"]
            conf = d["confidence"]
            color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        out.write(frame)

    cap.release()
    out.release()

    async def cleanup_async(path):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: shutil.rmtree(path, ignore_errors=True))

    return FileResponse(
        output_path,
        media_type="video/x-msvideo",
        filename="output.avi",
        background=lambda: cleanup_async(temp_dir)
    )

@router.post("/upload-video-report/")
async def upload_video_report(file: UploadFile = File(...), detailed: bool = False, db: AsyncSession = Depends(get_db)):
    """Принимает видео и возвращает JSON-отчёт о присутствии СИЗ и возможных нарушениях."""
    detector = detector_singleton.detector_instance
    if detector is None:
        return {"error": "Модель не загружена"}

    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else None

    class_occurrences = {name: 0 for name in detector.class_names}
    violations = []
    frame_details = [] if detailed else None

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)
        current_classes = set()
        frame_info = {"frame": frame_id, "detections": detections}

        for d in detections:
            cls_name = d["class_name"]
            class_occurrences[cls_name] += 1
            current_classes.add(cls_name)

        negative_classes = [cls for cls in current_classes if cls.startswith("NO-")]
        if negative_classes:
            violations.append({
                "frame": frame_id,
                "timestamp_sec": frame_id / fps if fps else None,
                "violations": negative_classes
            })

        if detailed:
            frame_details.append(frame_info)

        frame_id += 1

    cap.release()
    shutil.rmtree(temp_dir, ignore_errors=True)

    summary = []
    for name, count in class_occurrences.items():
        present = "yes" if count > 0 else "no"
        summary.append({
            "class": name,
            "detected_count": count,
            "presence": present,
            "is_violation": name.startswith("NO-") and count > 0
        })

    report = {
        "source": file.filename,
        "duration_seconds": duration,
        "frames_processed": frame_id,
        "summary": summary,
        "violations": violations,
    }
    if detailed:
        report["frame_details"] = frame_details

    return report