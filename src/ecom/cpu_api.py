"""Serve the actually trained tiny CPU Qwen3 policy. Not a useful 8B customer model."""

import os, time, uuid
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app):
    global model, tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(1)
    path = Path(os.getenv("MODEL_PATH", "artifacts/trl-smoke/merged"))
    if not path.exists():
        raise RuntimeError(
            "Run python -m ecom.trl_smoke first; no model is silently downloaded or substituted"
        )
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    yield


app = FastAPI(title="Tiny trained Qwen3 CPU integration backend", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class Completion(BaseModel):
    model: str = "tiny-qwen3"
    messages: list[Message] = Field(min_length=1, max_length=8)
    max_tokens: int = Field(default=16, ge=1, le=256)
    temperature: float = 0
    chat_template_kwargs: dict | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "random_tiny_qwen3_trained_cpu",
        "scope": "integration_only_not_customer_quality",
    }


@app.post("/v1/chat/completions")
def completion(body: Completion):
    import torch

    # The local WordLevel tokenizer has no conversational template. Match the smoke prompt vocabulary.
    prompt = body.messages[-1].content + " "
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        return_token_type_ids=False,
    )
    if encoded["input_ids"].shape[1] > 24:
        raise HTTPException(422, "Tiny demo context limit: 24 input tokens")
    allowed = min(body.max_tokens, 32)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=allowed,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    output = generated[0, encoded["input_ids"].shape[1] :]
    stopped = len(output) > 0 and int(output[-1]) == tokenizer.eos_token_id
    return {
        "id": "cpu-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.model,
        "scope": "random_tiny_qwen3_cpu_integration_only",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": tokenizer.decode(output, skip_special_tokens=True),
                },
                "finish_reason": "stop" if stopped else "length",
            }
        ],
        "usage": {
            "prompt_tokens": encoded["input_ids"].shape[1],
            "completion_tokens": len(output),
            "total_tokens": generated.shape[1],
        },
    }
