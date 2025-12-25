import time
import json
import uuid
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# Import Core
from core import Grok

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    API_KEY: Optional[str] = None
    GROK_PROXY: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

app = FastAPI(title="Grok OpenAI Wrapper")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "grok-3-auto"
    messages: List[ChatMessage]
    stream: bool = False

# --- Helper Logic ---

def build_prompt(messages: List[ChatMessage]) -> str:
    """
    Xử lý message list chuẩn OpenAI thành prompt đơn string cho Grok.
    Tự động ghép System Prompt vào User Prompt để tránh lỗi mất ngữ cảnh.
    """
    system_prompt = ""
    user_prompt = ""

    # Duyệt ngược để lấy tin nhắn mới nhất
    # Lưu ý: Grok wrapper stateless nên ta ưu tiên lấy cặp câu hỏi cuối cùng
    for msg in reversed(messages):
        if msg.role == "user" and not user_prompt:
            user_prompt = msg.content
        elif msg.role == "system" and not system_prompt:
            system_prompt = msg.content
        
        # Nếu đã có cả 2 thì dừng loop để tối ưu
        if user_prompt and system_prompt:
            break
    
    # Logic ghép: Nếu có system, đưa lên đầu làm context
    if system_prompt and user_prompt:
        return f"Instructions: {system_prompt}\n\nUser Query: {user_prompt}"
    
    if user_prompt:
        return user_prompt
        
    # Fallback nếu chỉ có system (ít gặp, nhưng để tránh crash)
    return system_prompt if system_prompt else "Hello"

def format_chunk(id: str, model: str, content: str, finish_reason: str = None) -> str:
    chunk = {
        "id": id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content} if content else {}, "finish_reason": finish_reason}]
    }
    return f"data: {json.dumps(chunk)}\n\n"

# --- Endpoints ---

async def verify_api_key(authorization: str = Header(None)):
    if not settings.API_KEY: return True
    if not authorization or authorization.split(" ")[1] != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return True

@app.get("/v1/models")
async def list_models(api_key: bool = Depends(verify_api_key)):
    return {"object": "list", "data": [{"id": "grok-3-auto", "object": "model", "owned_by": "xai"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key: bool = Depends(verify_api_key)):
    try:
        # 1. Build Prompt (Đã xử lý loại bỏ role system)
        final_message = build_prompt(request.messages)
        logger.info(f"Incoming Request: {len(request.messages)} msgs -> Prompt len: {len(final_message)}")

        # 2. Init Client
        client = Grok(model=request.model, proxy=settings.GROK_PROXY)
        request_id = f"chatcmpl-{uuid.uuid4()}"

        # 3. Streaming Logic
        if request.stream:
            def event_stream():
                yield format_chunk(request_id, request.model, "")
                
                # Gọi hàm chat_stream
                # Lưu ý: core/grok.py cần yield chuỗi lỗi bắt đầu bằng 'Error:' nếu failed
                for token in client.chat_stream(final_message):
                    if token.startswith("Error:"):
                        logger.error(f"Grok Error Stream: {token}")
                        # Trả về lỗi rõ ràng cho Client thay vì [Error: ] rỗng
                        clean_error = token.replace("Error:", "").strip()
                        if not clean_error: clean_error = "Unknown upstream error (403/400)"
                        yield format_chunk(request_id, request.model, f"\n[Grok Error: {clean_error}]", "stop")
                        break
                    yield format_chunk(request_id, request.model, token)
                
                yield format_chunk(request_id, request.model, "", "stop")
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # 4. Non-Streaming Logic
        else:
            response = client.start_convo(final_message)
            if "error" in response:
                raise HTTPException(status_code=500, detail=response["error"])
            
            return {
                "id": request_id,
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": response.get("response", "")}, "finish_reason": "stop"}]
            }

    except Exception as e:
        logger.error(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host=settings.HOST, port=settings.PORT, workers=settings.WORKERS)