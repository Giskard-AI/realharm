# RealHarm 


## Installation

```
uv sync
```

## Environment Variables

```
MODAL_HF_SECRET=<name of huggingface secret on modal>

MISTRAL_API_KEY=<your mistral key>
LAKERA_GUARD_API_KEY=<your lakera guard key>
PERSPECTIVE_API_KEY=<your perspective key>
AZURE_CONTENT_SAFETY_ENDPOINT=<your azure content safety endpoint>
AZURE_CONTENT_SAFETY_KEY=<your azure content safety key>

OPENAI_API_KEY=<your openai key> # Requires access to GPT-4o and OAI Moderation API

GEMINI_API_KEY=<your gemini key>
OPENROUTER_API_KEY=<your openrouter key> # could be replaced with anthropic api key, need to change the code of the llm based moderator associated with Claude 3.7 to use Anthropic as a provider


```

