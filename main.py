"""
main.py
-------
App entry point. Registers routers and manages startup/shutdown lifecycle.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import API_TITLE, API_DESCRIPTION, API_VERSION
from app.routers import health_router, predict_router
from app.services import SalaryPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan — runs once at startup and once at shutdown ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading pipeline bundle ...")
    try:
        app.state.predictor = SalaryPredictor()
        logger.info("Model ready ✅")
    except FileNotFoundError as e:
        logger.error(str(e))
        app.state.predictor = None
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(predict_router)
