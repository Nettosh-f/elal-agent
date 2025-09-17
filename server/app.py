# app.py
import os, uuid, datetime, time, json
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from jose import jwt, JWTError
from redis.asyncio import Redis
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
# ----- config -------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in env or .env")

client = OpenAI(api_key=OPENAI_API_KEY)
DEV_PASSWORD = os.getenv("DEV_PASSWORD", "demo@local")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
USE_FAKE_REDIS = os.getenv("USE_FAKE_REDIS", "0") == "1"

redis: Redis | None = None
SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", "1800"))  # default 30m
LOG_DIR = os.getenv("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)

JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret")
JWT_ALG = "HS256"
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))


async def init_redis():
    global redis
    if USE_FAKE_REDIS:
        from fakeredis.aioredis import FakeRedis
        redis = FakeRedis(decode_responses=True)
        print("[redis] Using FakeRedis (dev)")
        return

    redis = Redis.from_url(REDIS_URL, decode_responses=True)  # assign first
    try:
        await redis.ping()
        print("[redis] Connected:", REDIS_URL)
    except Exception as e:
        print("[redis] Cannot connect to", REDIS_URL, "->", e)
        from fakeredis.aioredis import FakeRedis
        redis = FakeRedis(decode_responses=True)
        print("[redis] Falling back to FakeRedis (dev)")


# ------------ App ------------
app = FastAPI(title="Simple Chat (Redis TTL + JWT + Rate Limit)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:63342",
        "http://127.0.0.1:63342",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://apk-trainer.netlify.app",
    ],
    # Allow Netlify preview/prod sites:
    allow_origin_regex=r"^https://.*\.netlify\.app$|^https://.*\.netlify\.live$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------ Models ------------


class ChatIn(BaseModel):
    message: str


class StartResp(BaseModel):
    session_id: str


class ChatReq(BaseModel):
    session_id: str
    message: str


class DevSignin(BaseModel):
    user_id: str


def retrieve_context(query: str, k: int = 4) -> List[Dict]:
    """
    TODO: Replace with real embeddings-based retrieval.
    Return list of {"title": str, "text": str}.
    """
    return []  # stub


# ------------ Auth helpers ------------
class AuthedUser(BaseModel):
    sub: str  # user id / email
    ul: bool = False


def require_auth(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload  # contains sub and maybe ul
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(req: Request) -> AuthedUser:
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Invalid token: missing sub")
        ul = bool(payload.get("ul", False))
        return AuthedUser(sub=sub, ul=ul)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def enforce_rate_limit(user: AuthedUser):
    if getattr(user, "ul", False):
        return  # unlimited session: skip rate limit
    window = 60
    now_bucket = int(time.time() // window)
    key = f"rl:{user.sub}:{now_bucket}"
    n = await redis.incr(key)
    if n == 1:
        await redis.expire(key, window)
    if n > RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try later.")


@app.get("/api/auth/dev-token")
def dev_token(user_id: str = "demo@local"):
    import time
    ul = (user_id == DEV_PASSWORD)
    token = jwt.encode(
        {"sub": user_id, "iat": int(time.time()), "ul": ul},
        JWT_SECRET,
        algorithm=JWT_ALG
    )
    print("[dev-token] requested for:", user_id)
    print("[dev-token] issued ul=", ul)
    return {"token": token, "ul": ul}


# ------------ Session storage (Redis) ------------
# Keys:
# sess:{sid}:meta -> {"created_at": iso, "user": sub}
# sess:{sid}:messages -> Redis list of JSON messages [{role, content, ts}]

def sess_meta_key(sid: str) -> str: return f"sess:{sid}:meta"


def sess_msgs_key(sid: str) -> str: return f"sess:{sid}:messages"


async def touch_ttl(sid: str):
    r = await get_redis_client()
    await r.expire(sess_meta_key(sid), SESSION_TTL)
    await r.expire(sess_msgs_key(sid), SESSION_TTL)


async def append_msg(sid: str, msg: Dict[str, Any]):
    r = await get_redis_client()
    await r.rpush(sess_msgs_key(sid), json.dumps(msg))
    await touch_ttl(sid)


async def get_msgs(sid: str) -> List[Dict[str, Any]]:
    r = await get_redis_client()
    raw = await r.lrange(sess_msgs_key(sid), 0, -1)
    return [json.loads(x) for x in raw]


async def session_exists(sid: str) -> bool:
    r = await get_redis_client()
    return (await r.exists(sess_msgs_key(sid))) == 1


SYSTEM_PROMPT = "You are a concise, helpful assistant. Keep answers short unless asked."


# ------------ Endpoints ------------

@app.post("/api/conversations/{session_id}/messages")
async def send_message(session_id: str, body: ChatIn, auth: str = Depends(require_auth)):
    # Reuse existing /api/chat handler
    return await chat_endpoint(ChatRequest(session_id=session_id, message=body.message), auth)


@app.post("/api/auth/dev-token")
def dev_token_post(body: DevSignin):
    return dev_token(user_id=body.user_id)


@app.post("/api/conversations", response_model=StartResp)
async def start_conversation(user: AuthedUser = Depends(get_current_user)):
    await enforce_rate_limit(user)
    sid = str(uuid.uuid4())
    created = datetime.datetime.utcnow().isoformat()
    await redis.hset(sess_meta_key(sid), mapping={"created_at": created, "user": user.sub})
    await redis.expire(sess_meta_key(sid), SESSION_TTL)
    await append_msg(sid, {"role": "system", "content": SYSTEM_PROMPT, "ts": ts()})
    return {"session_id": sid}


@app.post("/api/chat")
async def chat(req: ChatReq, user: AuthedUser = Depends(get_current_user)):
    await enforce_rate_limit(user)
    sid = req.session_id
    if not await session_exists(sid):
        raise HTTPException(status_code=404, detail="Unknown or expired session_id")

    # Append user message
    await append_msg(sid, {"role": "user", "content": req.message, "ts": ts()})

    # Build messages for the model (system + recent turns)
    history = await get_msgs(sid)
    system = [m for m in history if m["role"] == "system"][:1]
    recent = [m for m in history if m["role"] != "system"][-16:]
    messages = [
        {"role": m["role"], "content": m["content"]} for m in (system + recent)
    ]

    async def agen():
        buffer: List[str] = []

        def token_iter():
            with client.responses.stream(
                    model="gpt-4o-mini",
                    input=messages,
                    temperature=0.3,
            ) as stream:
                for event in stream:
                    yield event

        for event in token_iter():
            if event.type == "response.output_text.delta":
                chunk = event.delta
                buffer.append(chunk)
                yield chunk
            elif event.type == "response.completed":
                assistant_text = "".join(buffer)
                await append_msg(sid, {"role": "assistant", "content": assistant_text, "ts": ts()})
            elif event.type == "response.error":
                yield "\n[Error] " + str(event.error)

    return StreamingResponse(agen(), media_type="text/plain")


@app.post("/api/conversations/{session_id}/end")
async def end_conversation(session_id: str, user: AuthedUser = Depends(get_current_user)):
    await enforce_rate_limit(user)
    msgs = await get_msgs(session_id)
    meta = await redis.hgetall(sess_meta_key(session_id))
    if not msgs:
        raise HTTPException(status_code=404, detail="Unknown or already ended session")

    path = os.path.join(LOG_DIR, f"{session_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_log(meta, msgs))

    await redis.delete(sess_msgs_key(session_id))
    await redis.delete(sess_meta_key(session_id))

    return {"status": "ok", "log_path": path}


# ------------ Helpers ------------


@app.on_event("startup")
async def startup_event():
    print("[startup] init redis")
    await init_redis()


async def get_redis_client() -> Redis:
    global redis
    if redis is None:
        await init_redis()
    if redis is None:
        raise HTTPException(status_code=503, detail="Redis not initialized")
    return redis


def ts():
    return datetime.datetime.utcnow().isoformat()


def format_log(meta: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
    lines = [f"created_at: {meta.get('created_at', '')}"]
    if user := meta.get('user'): lines.append(f"user: {user}")
    lines.append("transcript:\n")
    for m in messages:
        lines.append(f"[{m['ts']}] {m['role'].upper()}: {m['content']}")
    return "\n".join(lines) + "\n"
