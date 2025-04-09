from pathlib import Path

import modal

HF_MODEL_DIR = Path("/models/hf")

app = modal.App("rh-realharm-bench")

volume = modal.Volume.from_name("rh-model-weights", create_if_missing=True)

inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers",
        "torch",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_transfer]",
        "llm-guard",
    )
    .env({"HF_HUB_CACHE": str(HF_MODEL_DIR), "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

inference_image_with_source = inference_image.add_local_python_source("run_benchmark")
