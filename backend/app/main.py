from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Engine

from app.config import Settings, get_settings
from app.db import make_engine, make_session_factory
from app.routers import admin, auth, picklists, pit, prescout, scouting, vision
from app.security import RateLimiter


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or get_settings()  # raises on weak prod config -> refuses to start
    app = FastAPI(
        title="FRC Scout AI backend",
        version="0.1.0",
        # No interactive docs / live schema: contract is exported to openapi.json.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.engine = engine or make_engine(settings.database_url)
    app.state.session_factory = make_session_factory(app.state.engine)
    app.state.limiter = RateLimiter(settings.login_max_failures, settings.login_window_seconds)

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
            allow_headers=["Content-Type"],
        )
    app.include_router(auth.router)
    app.include_router(admin.router)
    for r in (scouting, pit, picklists, prescout, vision):
        app.include_router(r.router)
    return app


def app_factory() -> FastAPI:
    """uvicorn --factory entry point."""
    return create_app()
