"""FastAPI application for the history book chat interface."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import agent, books, chat


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="History Book Chat API",
        description="API for RAG-based chat with historical documents",
        version="0.1.0",
    )

    # Configure CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # React dev server
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat.router, prefix="/api")
    app.include_router(agent.router, prefix="/api")
    app.include_router(books.router, prefix="/api")

    return app


app = create_app()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "History Book Chat API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
