import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE = "https://api.openai.com"
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "iqlytics-internal-key")

# Change models here only — Django never needs to know
MODEL_MAP = {
    "fast":   os.environ.get("FAST_MODEL", "gpt-4o-mini"),
    "smart":  os.environ.get("SMART_MODEL", "gpt-4o"),
}


def _verify_key(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if auth.replace("Bearer ", "").strip() != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid key")


def _resolve_model(body: dict) -> dict:
    requested = body.get("model", "smart")
    body["model"] = MODEL_MAP.get(requested, requested)
    return body


@app.get("/health")
def health():
    return {"status": "ok", "service": "iqlytics-llm-proxy"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _verify_key(request)
    body = _resolve_model(await request.json())
    stream = body.get("stream", False)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    if stream:
        async def stream_response():
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{OPENAI_BASE}/v1/chat/completions",
                    json=body,
                    headers=headers,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{OPENAI_BASE}/v1/chat/completions",
            json=body,
            headers=headers,
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return JSONResponse(content=resp.json())


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    _verify_key(request)
    body = _resolve_model(await request.json())

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{OPENAI_BASE}/v1/embeddings",
            json=body,
            headers=headers,
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return JSONResponse(content=resp.json())
