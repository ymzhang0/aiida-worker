"""Backward-compatible module that re-exports the FastAPI app from main."""

from __future__ import annotations

from main import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)
