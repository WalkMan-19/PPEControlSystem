import asyncio
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.detection.singleton import init_detector
from app.api.routes import router
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, init_detector)
    logger.info("Модель успешно загружена")
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(router)