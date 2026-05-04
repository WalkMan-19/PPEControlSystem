import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv()

DEBUG = os.getenv("DEBUG")

MODEL_PARAMS = {
    "model_path": os.path.join(BASE_DIR, "models", "best.pt"),
    "detector_type": "torch",
    "confidence_threshold": .1,
    "iou_threshold": .45,
    "input_width": 640,
    "input_height": 640,
    "class_names": [
        "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
        "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle"
    ],
}

DATABASE = {
    "HOST": os.getenv("DB_HOST", "localhost"),
    "PORT": int(os.getenv("DB_PORT", 5432)),
    "USER": os.getenv("DB_USER", "postgres"),
    "PASSWORD": os.getenv("DB_PASSWORD", "postgres"),
    "NAME": os.getenv("DB_NAME"),
}

DATABASE_URL = "postgresql+asyncpg://{}:{}@{}:{}/{}".format(
    DATABASE['USER'], DATABASE['PASSWORD'], DATABASE['HOST'], DATABASE['PORT'], DATABASE['NAME']
)

CLASS_COLORS = {
    'Hardhat': (0, 0, 255),
    'Mask': (0, 255, 0),
    'NO-Hardhat': (255, 0, 0),
    'NO-Mask': (0, 255, 255),
    'NO-Safety Vest': (255, 255, 0),
    'Person': (255, 0, 255),
    'Safety Cone': (0, 165, 255),
    'Safety Vest': (128, 0, 128),
    'machinery': (128, 128, 0),
    'vehicle': (203, 192, 255),
}
DEFAULT_COLOR = (255, 255, 255)
