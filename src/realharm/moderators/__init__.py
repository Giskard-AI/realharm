from .openai import OpenAIModerator
from .azure import AzureModerator
from .llm_guard import LLMGuardModerator
from .llama_guard import LlamaGuardModerator
from .langchain_eval import LangchainEvalModerator
from .lakera import LakeraModerator
from .perspective_api import PerspectiveModerator
from .mistral import MistralModerator
from .shield_gemma import ShieldGemmaModerator
from .ibm_granite import GraniteGuardModerator
from .llm_based import GPT4oModerator, Claude35Moderator, Claude37Moderator, GeminiModerator

__all__ = [
    "OpenAIModerator",
    "AzureModerator",
    "LLMGuardModerator",
    "LlamaGuardModerator",
    "LangchainEvalModerator",
    "LakeraModerator",
    "PerspectiveModerator",
    "MistralModerator",
    "GraniteGuardModerator",
]

MODERATOR_REPOSITORY = {
    "OpenAIModerator": OpenAIModerator,
    "AzureModerator": AzureModerator,
    "LLMGuardModerator": LLMGuardModerator,
    "LlamaGuardModerator": LlamaGuardModerator,
    "LangchainEvalModerator": LangchainEvalModerator,
    "LakeraModerator": LakeraModerator,
    "PerspectiveModerator": PerspectiveModerator,
    "MistralModerator": MistralModerator,
    "ShieldGemmaModerator": ShieldGemmaModerator,
    "GraniteGuardModerator": GraniteGuardModerator,
    "GPT4OModerator": GPT4oModerator,
    "Claude35Moderator": Claude35Moderator,
    "Claude37Moderator": Claude37Moderator,
    "GeminiModerator": GeminiModerator,
}
