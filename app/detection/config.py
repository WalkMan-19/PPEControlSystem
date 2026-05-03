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

