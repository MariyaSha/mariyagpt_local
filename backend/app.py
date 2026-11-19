import os
import threading
from collections.abc import Generator
from datetime import datetime

import psycopg
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import (
    hf_hub_download,
    snapshot_download,
)
from pydantic import BaseModel
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)


load_dotenv()

app = FastAPI(title="MariyaGPT API")


DATABASE_URL = os.getenv("DATABASE_URL")

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://127.0.0.1:5000",
)


if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not configured."
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


runtime = {
    "name": None,
    "model": None,
}


MODEL_CATALOG = {
    "1.5b": [
        {
            "id": "Qwen/Qwen2-1.5B-Instruct",
            "name": "Qwen2 1.5B Instruct",
        },
        {
            "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "name": "TinyLlama 1.1B Chat",
        },
    ],

    "3b": [
        {
            "id": "Qwen/Qwen2.5-3B-Instruct",
            "name": "Qwen2.5 3B Instruct",
        },
        {
            "id": "nvidia/Nemotron-Mini-4B-Instruct",
            "name": "NVIDIA Nemotron Mini 4B Instruct",
        },
    ],

    "7b": [
        {
            "id": "Qwen/Qwen2.5-7B-Instruct",
            "name": "Qwen2.5 7B Instruct",
        },
        {
            "id": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
            "name": "NVIDIA Llama 3.1 Nemotron Nano 8B",
        },
    ],
}


MODEL_IGNORE_PATTERNS = [
    "*.bin",
    "*.onnx",
    "*.gguf",
    "*.h5",
    "*.msgpack",
    "*.ot",
]


loaded_model_id = None
loaded_model = None
loaded_tokenizer = None

model_lock = threading.Lock()


download_lock = threading.Lock()

download_state = {
    "model": None,
    "status": "idle",
    "downloaded_bytes": 0,
    "total_bytes": 0,
    "error": None,
}


def detect_hardware():
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(
            0
        )

        vram_gb = (
            properties.total_memory
            / (1024 ** 3)
        )

        if vram_gb >= 12:
            model_tier = "7b"
        else:
            model_tier = "3b"

        return {
            "cuda": True,
            "device": properties.name,
            "vram_gb": round(vram_gb, 1),
            "model_tier": model_tier,
        }

    return {
        "cuda": False,
        "device": "CPU",
        "vram_gb": 0,
        "model_tier": "1.5b",
    }


def get_available_models():
    hardware_info = detect_hardware()

    return MODEL_CATALOG[
        hardware_info["model_tier"]
    ]


def get_available_model_ids():
    return {
        item["id"]
        for item in get_available_models()
    }


def bytes_to_gb(
    value: int,
) -> float:
    return round(
        value / (1024 ** 3),
        2,
    )


def get_model_download_info(
    model_id: str,
):
    if (
        model_id
        not in get_available_model_ids()
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "The selected model is not "
                "available for this hardware."
            ),
        )

    try:
        files = snapshot_download(
            repo_id=model_id,
            dry_run=True,
            ignore_patterns=(
                MODEL_IGNORE_PATTERNS
            ),
        )

    except Exception as error:
        print(
            "Model metadata error:",
            repr(error),
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Could not retrieve model "
                "information from Hugging Face."
            ),
        )

    total_bytes = sum(
        file.file_size
        for file in files
    )

    remaining_bytes = sum(
        file.file_size
        for file in files
        if not file.is_cached
    )

    cached_bytes = (
        total_bytes
        - remaining_bytes
    )

    downloaded = (
        remaining_bytes == 0
    )

    return {
        "model": model_id,
        "total_bytes": total_bytes,
        "cached_bytes": cached_bytes,
        "remaining_bytes":
            remaining_bytes,
        "size_gb": bytes_to_gb(
            total_bytes
        ),
        "downloaded": downloaded,
        "files": files,
    }


def unload_model():
    global loaded_model_id
    global loaded_model
    global loaded_tokenizer

    loaded_model_id = None
    loaded_model = None
    loaded_tokenizer = None

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()

        except Exception as error:
            print(
                "CUDA cache cleanup error:",
                repr(error),
                flush=True,
            )


class DownloadProgress(tqdm):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.base_downloaded = (
            download_state[
                "downloaded_bytes"
            ]
        )

    def update(
        self,
        amount=1,
    ):
        result = super().update(
            amount
        )

        if self.total:
            download_state[
                "downloaded_bytes"
            ] = min(
                self.base_downloaded
                + int(self.n),
                download_state[
                    "total_bytes"
                ],
            )

        return result


def run_model_download(
    model_id: str,
):
    try:
        info = get_model_download_info(
            model_id
        )

        download_state["model"] = (
            model_id
        )

        download_state["status"] = (
            "downloading"
        )

        download_state[
            "downloaded_bytes"
        ] = info["cached_bytes"]

        download_state[
            "total_bytes"
        ] = info["total_bytes"]

        download_state["error"] = None

        for file in info["files"]:
            if file.is_cached:
                continue

            before_file = download_state[
                "downloaded_bytes"
            ]

            try:
                hf_hub_download(
                    repo_id=model_id,
                    filename=file.filename,
                    tqdm_class=(
                        DownloadProgress
                    ),
                )

            except Exception:
                download_state[
                    "downloaded_bytes"
                ] = before_file

                raise

            download_state[
                "downloaded_bytes"
            ] = min(
                before_file
                + file.file_size,
                download_state[
                    "total_bytes"
                ],
            )

        final_info = (
            get_model_download_info(
                model_id
            )
        )

        if not final_info[
            "downloaded"
        ]:
            raise RuntimeError(
                "Model download did "
                "not complete."
            )

        download_state[
            "downloaded_bytes"
        ] = download_state[
            "total_bytes"
        ]

        download_state["status"] = (
            "complete"
        )

    except Exception as error:
        print(
            "Model download error:",
            repr(error),
            flush=True,
        )

        download_state["status"] = (
            "error"
        )

        download_state["error"] = str(
            error
        )


def load_local_model(
    model_id: str,
):
    global loaded_model_id
    global loaded_model
    global loaded_tokenizer

    with model_lock:
        if (
            loaded_model_id == model_id
            and loaded_model is not None
            and loaded_tokenizer is not None
        ):
            return (
                loaded_tokenizer,
                loaded_model,
            )

        download_info = (
            get_model_download_info(
                model_id
            )
        )

        if not download_info[
            "downloaded"
        ]:
            raise RuntimeError(
                "The selected model has "
                "not been downloaded yet."
            )

        unload_model()

        hardware_info = (
            detect_hardware()
        )

        print(
            "Loading model:",
            model_id,
            flush=True,
        )

        print(
            "Hardware:",
            hardware_info,
            flush=True,
        )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_id,
                local_files_only=True,
            )
        )

        if hardware_info["cuda"]:
            compute_dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )

            print(
                "CUDA compute dtype:",
                compute_dtype,
                flush=True,
            )

            quantization_config = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=(
                        compute_dtype
                    ),
                )
            )

            model = (
                AutoModelForCausalLM
                .from_pretrained(
                    model_id,
                    local_files_only=True,
                    device_map="auto",
                    quantization_config=(
                        quantization_config
                    ),
                    dtype=compute_dtype,
                    low_cpu_mem_usage=True,
                )
            )

        else:
            model = (
                AutoModelForCausalLM
                .from_pretrained(
                    model_id,
                    local_files_only=True,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
            )

            model.to("cpu")

        model.eval()

        loaded_model_id = model_id
        loaded_model = model
        loaded_tokenizer = tokenizer

        print(
            "Model loaded:",
            model_id,
            flush=True,
        )

        return (
            loaded_tokenizer,
            loaded_model,
        )


class SetupRequest(BaseModel):
    name: str
    model: str


class ModelRequest(BaseModel):
    model: str


class MessageRequest(BaseModel):
    message: str


class ConversationRequest(BaseModel):
    title: str


def create_conversation(
    title: str,
):
    with psycopg.connect(
        DATABASE_URL
    ) as connection:

        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO conversations (title)
                VALUES (%s)
                RETURNING
                    id,
                    title,
                    created_at;
                """,
                (title,),
            )

            return cursor.fetchone()


def get_conversation(
    conversation_id: int,
):
    with psycopg.connect(
        DATABASE_URL
    ) as connection:

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    title,
                    created_at
                FROM conversations
                WHERE id = %s;
                """,
                (conversation_id,),
            )

            return cursor.fetchone()


def get_conversations():
    with psycopg.connect(
        DATABASE_URL
    ) as connection:

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    title,
                    created_at
                FROM conversations
                ORDER BY
                    created_at DESC,
                    id DESC;
                """
            )

            return cursor.fetchall()


def save_message(
    conversation_id: int,
    role: str,
    content: str,
):
    with psycopg.connect(
        DATABASE_URL
    ) as connection:

        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO messages (
                    conversation_id,
                    role,
                    content
                )
                VALUES (%s, %s, %s)
                RETURNING id;
                """,
                (
                    conversation_id,
                    role,
                    content,
                ),
            )

            return cursor.fetchone()[0]


def delete_message(
    message_id: int,
):
    with psycopg.connect(
        DATABASE_URL
    ) as connection:

        with connection.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM messages
                WHERE id = %s;
                """,
                (message_id,),
            )


def build_chat_messages(
    stored_messages,
):
    chat_messages = []

    for row in stored_messages:
        role = row[0]
        content = row[1]

        if role not in {
            "user",
            "assistant",
        }:
            continue

        if (
            chat_messages
            and chat_messages[-1]["role"]
            == role
        ):
            chat_messages[-1] = {
                "role": role,
                "content": content,
            }

            continue

        chat_messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    return chat_messages


def get_messages(
    conversation_id: int,
):
    with psycopg.connect(
        DATABASE_URL
    ) as connection:

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    role,
                    content,
                    created_at
                FROM messages
                WHERE conversation_id = %s
                ORDER BY created_at, id;
                """,
                (conversation_id,),
            )

            return cursor.fetchall()


def serialize_datetime(
    value: datetime,
) -> str:
    return value.isoformat()


@app.get("/")
def root():
    return {
        "message":
            "MariyaGPT API is running"
    }


@app.get("/api/hardware")
def hardware():
    hardware_info = (
        detect_hardware()
    )

    models = []

    for model in get_available_models():
        try:
            model_info = (
                get_model_download_info(
                    model["id"]
                )
            )

            models.append(
                {
                    "id": model["id"],
                    "name": model["name"],
                    "size_gb":
                        model_info[
                            "size_gb"
                        ],
                    "downloaded":
                        model_info[
                            "downloaded"
                        ],
                }
            )

        except Exception as error:
            print(
                "Skipping unavailable model:",
                model["id"],
                repr(error),
                flush=True,
            )

    hardware_info["models"] = (
        models
    )

    return hardware_info


@app.get("/api/health")
def health():
    return {
        "status": "healthy"
    }


@app.post(
    "/api/models/status"
)
def model_status(
    data: ModelRequest,
):
    info = get_model_download_info(
        data.model
    )

    return {
        "model": data.model,
        "size_gb":
            info["size_gb"],
        "downloaded":
            info["downloaded"],
    }


@app.post(
    "/api/models/download"
)
def start_model_download(
    data: ModelRequest,
):
    model_id = data.model.strip()

    if (
        model_id
        not in get_available_model_ids()
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "The selected model is "
                "not available for this "
                "hardware."
            ),
        )

    with download_lock:
        if (
            download_state["status"]
            == "downloading"
        ):
            if (
                download_state["model"]
                == model_id
            ):
                return {
                    "started": False,
                    "status":
                        "downloading",
                }

            raise HTTPException(
                status_code=409,
                detail=(
                    "Another model is "
                    "currently downloading."
                ),
            )

        info = get_model_download_info(
            model_id
        )

        if info["downloaded"]:
            download_state[
                "model"
            ] = model_id

            download_state[
                "status"
            ] = "complete"

            download_state[
                "downloaded_bytes"
            ] = info["total_bytes"]

            download_state[
                "total_bytes"
            ] = info["total_bytes"]

            download_state[
                "error"
            ] = None

            return {
                "started": False,
                "status": "complete",
            }

        download_state[
            "model"
        ] = model_id

        download_state[
            "status"
        ] = "starting"

        download_state[
            "downloaded_bytes"
        ] = info["cached_bytes"]

        download_state[
            "total_bytes"
        ] = info["total_bytes"]

        download_state[
            "error"
        ] = None

        thread = threading.Thread(
            target=run_model_download,
            args=(model_id,),
            daemon=True,
        )

        thread.start()

    return {
        "started": True,
        "status": "starting",
    }


@app.get(
    "/api/models/download-status"
)
def model_download_status():
    total_bytes = (
        download_state[
            "total_bytes"
        ]
    )

    downloaded_bytes = (
        download_state[
            "downloaded_bytes"
        ]
    )

    if total_bytes > 0:
        percent = round(
            (
                downloaded_bytes
                / total_bytes
            )
            * 100,
            1,
        )
    else:
        percent = 0

    return {
        "model":
            download_state["model"],
        "status":
            download_state["status"],
        "downloaded_bytes":
            downloaded_bytes,
        "total_bytes":
            total_bytes,
        "downloaded_gb":
            bytes_to_gb(
                downloaded_bytes
            ),
        "total_gb":
            bytes_to_gb(
                total_bytes
            ),
        "percent":
            percent,
        "error":
            download_state["error"],
    }


@app.post("/api/setup")
def setup(
    data: SetupRequest,
):
    name = data.name.strip()
    model = data.model.strip()

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Name is required.",
        )

    if not model:
        raise HTTPException(
            status_code=400,
            detail="Model is required.",
        )

    if (
        model
        not in get_available_model_ids()
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "The selected model is "
                "not available for this "
                "hardware."
            ),
        )

    download_info = (
        get_model_download_info(
            model
        )
    )

    if not download_info[
        "downloaded"
    ]:
        raise HTTPException(
            status_code=400,
            detail=(
                "Download the selected "
                "model before starting "
                "MariyaGPT."
            ),
        )

    if (
        runtime["model"]
        and runtime["model"] != model
    ):
        unload_model()

    runtime["name"] = name
    runtime["model"] = model

    return {
        "ready": True,
        "name": name,
        "model": model,
    }


@app.get("/api/setup")
def get_setup():
    return {
        "ready": bool(
            runtime["name"]
            and runtime["model"]
        ),
        "name": runtime["name"],
        "model": runtime["model"],
    }


@app.get("/api/conversations")
def list_conversations():
    rows = get_conversations()

    conversations = []

    for row in rows:
        conversations.append(
            {
                "id": row[0],
                "title": row[1],
                "created_at":
                    serialize_datetime(
                        row[2]
                    ),
            }
        )

    return {
        "conversations":
            conversations
    }


@app.post("/api/conversations")
def new_conversation(
    data: ConversationRequest,
):
    title = data.title.strip()

    if not title:
        raise HTTPException(
            status_code=400,
            detail=(
                "Conversation title "
                "is required."
            ),
        )

    conversation = (
        create_conversation(
            title
        )
    )

    return {
        "conversation": {
            "id":
                conversation[0],
            "title":
                conversation[1],
            "created_at":
                serialize_datetime(
                    conversation[2]
                ),
        }
    }


@app.get(
    "/api/conversations/"
    "{conversation_id}/messages"
)
def list_messages(
    conversation_id: int,
):
    conversation = get_conversation(
        conversation_id
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation not found."
            ),
        )

    rows = get_messages(
        conversation_id
    )

    messages = []

    for row in rows:
        messages.append(
            {
                "role": row[0],
                "content": row[1],
                "created_at":
                    serialize_datetime(
                        row[2]
                    ),
            }
        )

    return {
        "conversation": {
            "id":
                conversation[0],
            "title":
                conversation[1],
            "created_at":
                serialize_datetime(
                    conversation[2]
                ),
        },
        "messages": messages,
    }


@app.post(
    "/api/conversations/"
    "{conversation_id}/messages"
)
def stream_message(
    conversation_id: int,
    data: MessageRequest,
):
    if not runtime["model"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "MariyaGPT is not "
                "configured yet."
            ),
        )

    conversation = get_conversation(
        conversation_id
    )

    if not conversation:
        raise HTTPException(
            status_code=404,
            detail=(
                "Conversation not found."
            ),
        )

    user_message = (
        data.message.strip()
    )

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail=(
                "Please enter a message."
            ),
        )

    user_message_id = save_message(
        conversation_id,
        "user",
        user_message,
    )

    stored_messages = (
        get_messages(
            conversation_id
        )
    )

    chat_messages = (
        build_chat_messages(
            stored_messages
        )
    )

    def generate() -> (
        Generator[str, None, None]
    ):
        complete_response = ""

        try:
            tokenizer, model = (
                load_local_model(
                    runtime["model"]
                )
            )

            recent_messages = (
                chat_messages[-20:]
            )

            prompt = (
                tokenizer
                .apply_chat_template(
                    recent_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
            )

            model_device = next(
                model.parameters()
            ).device

            inputs = {
                key: value.to(
                    model_device
                )
                for key, value
                in inputs.items()
            }

            streamer = (
                TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
            )

            generation_error = []

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = (
                    tokenizer.eos_token_id
                )

            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": 1024,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "pad_token_id":
                    tokenizer.pad_token_id,
                "eos_token_id":
                    tokenizer.eos_token_id,
            }

            def run_generation():
                try:
                    model.generate(
                        **generation_kwargs
                    )

                except Exception as error:
                    generation_error.append(
                        error
                    )

                    streamer.on_finalized_text(
                        "",
                        stream_end=True,
                    )

            generation_thread = (
                threading.Thread(
                    target=(
                        run_generation
                    ),
                    daemon=True,
                )
            )

            generation_thread.start()

            for text in streamer:
                if text:
                    complete_response += (
                        text
                    )

                    yield text

            generation_thread.join()

            if generation_error:
                raise generation_error[0]

            if complete_response:
                save_message(
                    conversation_id,
                    "assistant",
                    complete_response,
                )

        except Exception as error:
            print(
                "Local model error:",
                repr(error),
                flush=True,
            )

            try:
                delete_message(
                    user_message_id
                )

            except Exception as cleanup_error:
                print(
                    "Failed to remove "
                    "unsuccessful user message:",
                    repr(cleanup_error),
                    flush=True,
                )

            yield (
                "\n\n"
                "MariyaGPT could not "
                "complete the response. "
                "Please try again."
            )

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/reset")
def reset_setup(request: Request):
    # A custom header requires browser preflight; CORS only trusts our frontend.
    # Check before clearing configuration or unloading the model.
    if request.headers.get("X-Requested-With") != "XMLHttpRequest":
        raise HTTPException(status_code=403, detail="Reset requires an AJAX request.")

    runtime["name"] = None
    runtime["model"] = None

    unload_model()

    return {
        "ready": False
    }