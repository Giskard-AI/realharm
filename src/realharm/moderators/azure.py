"""Azure Content Safety API"""
import os
import requests

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential

from realharm.models import ModerationOutput
from realharm.moderators.base import BaseModerator

class AzureModerator(BaseModerator):
    def __init__(self, endpoint: str | None = None, key: str | None = None):
        self._endpoint = endpoint or os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
        self._credential = AzureKeyCredential(
            key or os.getenv("AZURE_CONTENT_SAFETY_KEY")
        )
        self.client = ContentSafetyClient(self._endpoint, self._credential)

    def check(self, conversation: list[dict[str, str]], is_sample_safe:bool, **kwargs) -> ModerationOutput:
        for msg in conversation:
            if is_sample_safe and msg["role"] == "user":
                continue
            
            if not is_sample_safe and msg["role"] == "user":
                if self.is_prompt_unsafe(msg["content"]):
                    return ModerationOutput(safe=False, categories=["unsafe"])

            response = self.client.analyze_text(AnalyzeTextOptions(text=msg["content"]))
            categories = [
                f"{item.category}:L{item.severity}"
                for item in response.categories_analysis
                if item.severity > 0
            ]

            if categories:
                return ModerationOutput(safe=False, categories=categories)

        return ModerationOutput(safe=True, categories=[])


    def is_prompt_unsafe(self, prompt: str) -> bool:
        # Define the endpoint URL
        url = f"{self._endpoint}/contentsafety/text:shieldPrompt?api-version=2024-09-01"
        # Set up the headers
        headers = {
            'Ocp-Apim-Subscription-Key': self._credential.key,
            'Content-Type': 'application/json'
        }

        # Create the payload
        payload = {
            "userPrompt": prompt
        }

        # Make the POST request
        response = requests.post(url, headers=headers, json=payload)

        # Check the response status code
        if response.status_code == 200:
            print("Request was successful.")
            # Print the response content
            result = response.json()
            return result["userPromptAnalysis"]["attackDetected"]
        else:
            print(f"Request failed with status code {response.status_code}")
            print(response.text)