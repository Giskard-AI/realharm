import modal
import os

from realharm.modal.app import app, volume, inference_image_with_source, HF_MODEL_DIR

MODAL_HF_SECRET = os.getenv("MODAL_HF_SECRET")

@app.cls(
    volumes={HF_MODEL_DIR: volume},
    gpu="l4",
    image=inference_image_with_source,
    secrets=[modal.Secret.from_name(MODAL_HF_SECRET)],
)
class PromptGuardModel:
    def __init__(self, repo_id: str = "meta-llama/Prompt-Guard-86M"):
        self.repo_id = repo_id
        self.model = None
        self.tokenizer = None

    # @modal.build()
    # def build(self):
    #     from huggingface_hub import snapshot_download

    #     snapshot_download(self.repo_id)

    @modal.enter()
    def enter(self):
        import torch
        from transformers import pipeline

        self.classifier = pipeline("text-classification", model="meta-llama/Prompt-Guard-86M")


    @modal.method()
    def run(self, message: str):

        if self.classifier is None:
            raise RuntimeError("Model not initialized!")

        # 512 x 4 characters = 2048 characters, let's take a little margin 

        # Split long messages into 1500 char sequences
        sequences = [message[i:i+1500] for i in range(0, len(message), 1500)]
        # Run classifier on each sequence and take most concerning result
        results = self.classifier(sequences)
        # Return result with highest score
        return any(result["label"] == "JAILBREAK" for result in results)