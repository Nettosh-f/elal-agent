# app.py
import datetime
import json
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from jose import jwt, JWTError
from openai import OpenAI
from pydantic import BaseModel
from redis.asyncio import Redis

from prompt import PROMPT

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
# ----- config -------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in env or .env")

client = OpenAI(api_key=OPENAI_API_KEY)
DEV_PASSWORD = os.getenv("DEV_PASSWORD", "demo@local")
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
USE_FAKE_REDIS = os.getenv("USE_FAKE_REDIS", "0") == "1"
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
redis: Redis | None = None
SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", "1800"))  # default 30m
LOG_DIR = os.getenv("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)
ASSISTANT_ID = os.getenv("ASSISTANT_ID")
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


class StartIn(BaseModel):
    mode: str = "qa"  # "qa" or "trainer"


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


SYSTEM_PROMPT = PROMPT


# ------------ Endpoints ------------

@app.post("/api/conversations/{session_id}/messages")
async def send_message(session_id: str, body: ChatIn, user: AuthedUser = Depends(get_current_user)):
    return await chat(ChatReq(session_id=session_id, message=body.message), user)  # reuse


@app.post("/api/auth/dev-token")
def dev_token_post(body: DevSignin):
    return dev_token(user_id=body.user_id)


@app.post("/api/conversations", response_model=StartResp)
async def start_conversation(body: StartIn, user: AuthedUser = Depends(get_current_user)):
    await enforce_rate_limit(user)
    sid = str(uuid.uuid4())
    created = datetime.datetime.utcnow().isoformat()

    # choose prompt based on mode
    system_prompt = PROMPT if body.mode == "trainer" else QA_PROMPT

    await redis.hset(sess_meta_key(sid), mapping={
        "created_at": created,
        "user": user.sub,
        "mode": body.mode
    })
    await redis.expire(sess_meta_key(sid), SESSION_TTL)

    # store system prompt into the session’s message log (unchanged flow)
    await append_msg(sid, {"role": "system", "content": system_prompt, "ts": ts()})
    return {"session_id": sid}


@app.post("/api/chat")
async def chat(req: ChatReq, user: AuthedUser = Depends(get_current_user)):
    await enforce_rate_limit(user)

    # Validate the session and store user turn in YOUR transcript
    if not await session_exists(req.session_id):
        raise HTTPException(status_code=404, detail="Unknown or expired session_id")
    await append_msg(req.session_id, {"role": "user", "content": req.message, "ts": ts()})

    # 1) Make sure we have an Assistant and a Thread for this session
    asst_id = ensure_assistant_id()
    thread_id = await get_thread_id_for_session(req.session_id)

    # 2) Add the user message to the OpenAI Thread
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=req.message,
    )

    # 3) Stream the Run
    async def agen():
        buffer = []
        # NOTE: runs.stream is a sync context manager; OK inside async generator
        with client.beta.threads.runs.stream(
            thread_id=thread_id,
            assistant_id=asst_id,
        ) as stream:
            for event in stream:
                et = getattr(event, "type", "")
                if et == "response.output_text.delta":
                    chunk = event.delta
                    buffer.append(chunk)
                    yield chunk
                elif et == "response.error":
                    yield f"\n[error] {event.error}"
                elif et == "response.completed":
                    # Persist assistant final text to YOUR transcript
                    final_text = stream.get_final_response().output_text
                    await append_msg(
                        req.session_id,
                        {"role": "assistant", "content": final_text, "ts": ts()}
                    )

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
def ensure_assistant_id() -> str:
    """
    Reuse ASSISTANT_ID if provided; otherwise create an Assistant wired
    to your vector store for retrieval via file_search.
    """
    global ASSISTANT_ID
    if ASSISTANT_ID:
        return ASSISTANT_ID
    if not VECTOR_STORE_ID:
        raise RuntimeError("VECTOR_STORE_ID is required for retrieval.")
    asst = client.beta.assistants.create(
        name="Pilot Trainer / Q&A",
        model="gpt-4o-mini",
        # Use your training prompt if you want it globally; or keep this blank and set per-session in messages
        # instructions=PROMPT,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [VECTOR_STORE_ID]}},
    )
    ASSISTANT_ID = asst.id
    print("[assistant] Created:", ASSISTANT_ID)
    return ASSISTANT_ID


async def build_messages_for_session(sid: str, new_user_msg: str) -> list[dict]:
    # verify session
    if not await session_exists(sid):
        raise HTTPException(status_code=404, detail="Unknown or expired session_id")

    # append new user turn into Redis transcript
    await append_msg(sid, {"role": "user", "content": new_user_msg, "ts": ts()})

    # rebuild messages for OpenAI (system first, then rest)
    msgs = await get_msgs(sid)
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


async def store_assistant_turn(sid: str, text: str):
    await append_msg(sid, {"role": "assistant", "content": text, "ts": ts()})


async def get_thread_id_for_session(sid: str) -> str:
    """Create or load the OpenAI Thread ID tied to this session."""
    r = await get_redis_client()
    tid = await r.hget(sess_meta_key(sid), "thread_id")
    if tid:
        return tid
    thread = client.beta.threads.create()
    tid = thread.id
    await r.hset(sess_meta_key(sid), mapping={"thread_id": tid})
    await r.expire(sess_meta_key(sid), SESSION_TTL)
    return tid



@app.on_event("startup")
async def startup_event():
    print("[startup] init redis")
    await init_redis()
    ensure_assistant_id()

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
