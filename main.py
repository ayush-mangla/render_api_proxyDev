import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENAI_BASE = "https://api.openai.com"
OPENROUTER_BASE = "https://openrouter.ai/api"
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "iqlytics-internal-key")

# Short aliases → full OpenRouter model IDs
# Set FAST_MODEL or SMART_MODEL in Render to any alias below (or a full model ID)
ALIASES = {
    "sonnet-4.6":     "anthropic/claude-sonnet-4.6",
    "deepseek-v4":    "deepseek/deepseek-v4-pro",
    "qwen-3.7":       "qwen/qwen3.7-max",
    "gemini-flash":   "google/gemini-3.5-flash",
    "minimax-m3":     "minimax/minimax-m3",
    "nemotron-ultra": "nvidia/nemotron-3-ultra-550b-a55b:free",
}


def _expand(val: str) -> str:
    return ALIASES.get(val, val)


# Resolved once at startup — change FAST_MODEL/SMART_MODEL in Render dashboard
MODEL_MAP = {
    "fast":  _expand(os.environ.get("FAST_MODEL", "gpt-4o-mini")),
    "smart": _expand(os.environ.get("SMART_MODEL", "gpt-4o")),
    **ALIASES,
}


def _verify_key(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")
    if auth.replace("Bearer ", "").strip() != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid key")


def _resolve_model(body: dict) -> tuple[dict, str, str]:
    """Resolves model alias and returns (body, base_url, api_key)."""
    requested = body.get("model", "smart")
    resolved = MODEL_MAP.get(requested, requested)
    body["model"] = resolved
    if "/" in resolved:
        # Ask OpenRouter to return the real $ cost in the usage object.
        # For streaming, this also makes the final chunk carry usage+cost.
        body.setdefault("usage", {"include": True})
        return body, OPENROUTER_BASE, OPENROUTER_API_KEY
    return body, OPENAI_BASE, OPENAI_API_KEY


@app.get("/health")
def health():
    return {"fast": MODEL_MAP["fast"], "smart": MODEL_MAP["smart"]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _verify_key(request)
    body, base, api_key = _resolve_model(await request.json())
    stream = body.get("stream", False)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if stream:
        async def stream_response():
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{base}/v1/chat/completions",
                    json=body,
                    headers=headers,
                ) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{base}/v1/chat/completions",
            json=body,
            headers=headers,
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return JSONResponse(content=resp.json())


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    _verify_key(request)
    body = await request.json()

    # Embeddings always route to OpenAI — OpenRouter does not support this endpoint
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
