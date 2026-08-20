# MariyaGPT Local

MariyaGPT Local is a self-contained AI chat application that runs local language models with Docker.

It provides a simple ChatGPT-style interface powered by locally downloaded Hugging Face models, with persistent conversation history stored in PostgreSQL.

The application automatically detects the available hardware and offers models appropriate for the system. NVIDIA GPUs can use larger GPU-accelerated models, while systems without a compatible GPU can run smaller models on the CPU.

The complete frontend, backend, database, and local model runtime are packaged inside a single Docker image.

No OpenRouter account or API key is required.

## Architecture

MariyaGPT Local uses three main application components packaged inside a single Docker container:

```text
Browser
   │
   ▼
Flask Frontend
   │
   ▼
FastAPI Backend
   ├──► Local AI Model
   │       └──► Hugging Face Model Cache
   │
   └──► PostgreSQL
           └──► Conversations and Messages
```

The main components are:

* **Flask** — serves the web interface.
* **FastAPI** — handles application logic, hardware detection, model downloads, and local inference.
* **PostgreSQL** — stores conversation titles and message history.
* **Hugging Face** — provides the local AI model files.
* **Transformers** — loads and runs the selected model.
* **PyTorch** — provides CPU and NVIDIA CUDA inference.
* **Docker** — packages the complete application and its dependencies into one image.

Only the Flask frontend is exposed to the host computer on port `5000`. The FastAPI backend and PostgreSQL database remain internal to the container.

## Requirements

To run MariyaGPT Local you need:

* Docker Desktop or Docker Engine
* Enough disk space for the selected AI model

You do not need to install Python, PostgreSQL, Flask, FastAPI, PyTorch, Transformers, or any other application dependencies manually.

### CPU

MariyaGPT Local can run without a dedicated GPU.

If CUDA is not available, the application automatically detects CPU mode and offers smaller models intended for CPU inference.

### NVIDIA GPU

For GPU acceleration you need:

* A compatible NVIDIA GPU
* Current NVIDIA drivers
* Docker configured with NVIDIA GPU support

The GPU must be made available to the container using:

```text
--gpus all
```

MariyaGPT Local detects the GPU and available VRAM automatically and offers models appropriate for the detected hardware.

## Available Models

The model selector changes automatically depending on the detected hardware.

### CPU

When CUDA is not available, MariyaGPT Local offers:

| Model | Approximate Size |
| --- | ---: |
| Qwen2 1.5B Instruct | ~3 GB |
| TinyLlama 1.1B Chat | ~2 GB |

### NVIDIA GPU with less than 12 GB VRAM

MariyaGPT Local offers:

| Model |
| --- |
| Qwen2.5 3B Instruct |
| NVIDIA Nemotron Mini 4B Instruct |

### NVIDIA GPU with 12 GB VRAM or more

MariyaGPT Local offers:

| Model | Approximate Size |
| --- | ---: |
| Qwen2.5 7B Instruct | ~14.2 GB |
| NVIDIA Llama 3.1 Nemotron Nano 8B | ~16 GB |

Exact download sizes are calculated by MariyaGPT from Hugging Face and displayed in the setup interface before downloading.

## Run with Docker

Pull the image from Docker Hub:

```bash
docker pull mariyasha/mariyagpt_local:latest
```

### Run with CPU

```bash
docker run -d \
  --name mariyagpt \
  -p 5000:5000 \
  -v mariyagpt_data:/var/lib/postgresql/data \
  -v mariyagpt_models:/root/.cache/huggingface \
  mariyasha/mariyagpt_local:latest
```

Then open:

```text
http://localhost:5000
```

### Run with an NVIDIA GPU

```bash
docker run -d \
  --gpus all \
  --name mariyagpt \
  -p 5000:5000 \
  -v mariyagpt_data:/var/lib/postgresql/data \
  -v mariyagpt_models:/root/.cache/huggingface \
  mariyasha/mariyagpt_local:latest
```

Then open:

```text
http://localhost:5000
```

## Hardware Detection

When MariyaGPT Local starts, the backend checks the hardware available inside the container.

If an NVIDIA CUDA GPU is available, the setup page displays the detected GPU and VRAM.

For example:

```text
Hardware detected
NVIDIA GeForce RTX 5090 (31.8 GB VRAM)
```

If CUDA is not available, MariyaGPT automatically switches to CPU mode.

The available model list is selected automatically according to the detected hardware.

## Downloading Models

Models are not downloaded automatically when the container starts.

After selecting a model, MariyaGPT displays its approximate download size and provides a **Download Model** button.

For example:

```text
Download Model (~14.20 GB)
```

The download progress is displayed directly in the setup interface.

The **Start Chatting** button remains disabled until the selected model has been downloaded successfully.

Once the download completes, the model is available for local inference.

## Persistent Model Storage

Downloaded AI models are stored in the Docker volume:

```text
mariyagpt_models
```

It is mounted inside the container at:

```text
/root/.cache/huggingface
```

Downloaded models remain available even if the MariyaGPT container is stopped, removed, rebuilt, or replaced.

For example:

```bash
docker stop mariyagpt
docker rm mariyagpt
```

You can then create another container using the same volume:

```bash
docker run -d \
  --gpus all \
  --name mariyagpt \
  -p 5000:5000 \
  -v mariyagpt_data:/var/lib/postgresql/data \
  -v mariyagpt_models:/root/.cache/huggingface \
  mariyasha/mariyagpt_local:latest
```

Previously downloaded models will still be available.

**You do not need to download a model again unless its cached files or the `mariyagpt_models` Docker volume are deleted.**

The same model volume can be reused between CPU and GPU containers.

## Persistent Conversations

Conversation titles and messages are stored in PostgreSQL.

The PostgreSQL data directory is persisted using:

```text
mariyagpt_data
```

This means conversation history survives container removal and replacement as long as the volume is preserved.

Both persistent resources are independent:

```text
mariyagpt_data
    └── conversations and messages

mariyagpt_models
    └── downloaded Hugging Face models
```

## Restart or Replace the Container

You can safely stop and remove the application container:

```bash
docker stop mariyagpt
docker rm mariyagpt
```

Then recreate it with the same volumes.

### GPU

```bash
docker run -d \
  --gpus all \
  --name mariyagpt \
  -p 5000:5000 \
  -v mariyagpt_data:/var/lib/postgresql/data \
  -v mariyagpt_models:/root/.cache/huggingface \
  mariyasha/mariyagpt_local:latest
```

### CPU

```bash
docker run -d \
  --name mariyagpt \
  -p 5000:5000 \
  -v mariyagpt_data:/var/lib/postgresql/data \
  -v mariyagpt_models:/root/.cache/huggingface \
  mariyasha/mariyagpt_local:latest
```

Your previous conversations and downloaded models will still be available.

## Delete Conversation Data

To permanently delete all stored conversations:

```bash
docker stop mariyagpt
docker rm mariyagpt
docker volume rm mariyagpt_data
```

This does not delete downloaded AI models.

## Delete Downloaded Models

To permanently delete all downloaded models:

```bash
docker stop mariyagpt
docker rm mariyagpt
docker volume rm mariyagpt_models
```

The models will need to be downloaded again the next time they are selected.

This does not delete conversations stored in `mariyagpt_data`.

## Completely Reset MariyaGPT

To delete both conversation history and downloaded models:

```bash
docker stop mariyagpt
docker rm mariyagpt
docker volume rm mariyagpt_data
docker volume rm mariyagpt_models
```

The next container will start with a fresh PostgreSQL database and an empty Hugging Face model cache.

## Build from Source

Clone the GitHub repository:

```bash
git clone https://github.com/MariyaSha/mariyagpt_local.git
cd mariyagpt_local
```

Build:

```bash
docker build -t mariyagpt_local .
```

### Run the locally built image with CPU

```bash
docker run -d \
  --name mariyagpt \
  -p 5000:5000 \
  -v mariyagpt_data:/var/lib/postgresql/data \
  -v mariyagpt_models:/root/.cache/huggingface \
  mariyagpt_local
```

### Run the locally built image with NVIDIA GPU acceleration

```bash
docker run -d \
  --gpus all \
  --name mariyagpt \
  -p 5000:5000 \
  -v mariyagpt_data:/var/lib/postgresql/data \
  -v mariyagpt_models:/root/.cache/huggingface \
  mariyagpt_local
```

Then open:

```text
http://localhost:5000
```

Developers can modify the Flask frontend, FastAPI backend, database schema, model catalog, styles, JavaScript, or local inference logic and rebuild the image using the same Docker commands.

## Project Structure

```text
mariyagpt_local/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── start.sh
├── supervisord.conf
│
├── backend/
│   ├── app.py
│   └── requirements.txt
│
├── database/
│   └── schema.sql
│
└── frontend/
    ├── app.py
    ├── requirements.txt
    ├── static/
    │   ├── chat.js
    │   ├── setup.js
    │   └── styles.css
    └── templates/
        ├── chat.html
        └── home.html
```

## License

MariyaGPT Local is released under the MIT License.

See the `LICENSE` file for details.