from typing import Any, AsyncIterator, Dict, List, Optional

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential

from ..config.settings import settings
from ..core.exceptions import LLMException
from ..core.interfaces import LLMInterface, StreamingChunk
from ..utils.logging import logger


class GitHubLLM(LLMInterface):
    """GitHub Models LLM implementation with override support"""

    def __init__(
        self,
        model_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
        max_tokens_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        try:
            self.client = ChatCompletionsClient(
                endpoint="https://models.github.ai/inference",
                credential=AzureKeyCredential(settings.github_token),
            )
            self.model = model_override or settings.llm_model
            self.temperature = temperature_override or settings.llm_temperature
            self.max_tokens = max_tokens_override or settings.llm_max_tokens

        except Exception as e:
            logger.error("GitHub LLM initialization failed", error=str(e))
            raise LLMException(f"Failed to initialize GitHub LLM: {e}")

    async def generate_stream(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[StreamingChunk]:
        """Generate text from prompt with streaming"""
        try:
            response = self.client.complete(
                stream=True,
                messages=[UserMessage(prompt)],
                model=self.model,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                # model_extras={"stream_options": {"include_usage": True}},
            )
            for update in response:
                chunk_content = ""
                usage_info = None
                is_complete = False

                if update.choices and update.choices[0].delta:
                    chunk_content = update.choices[0].delta.content or ""

                if update.usage:
                    usage_info = {
                        "completion_tokens": getattr(
                            update.usage, "completion_tokens", 0
                        ),
                        "prompt_tokens": getattr(update.usage, "prompt_tokens", 0),
                        "total_tokens": getattr(update.usage, "total_tokens", 0),
                    }
                    is_complete = True

                yield StreamingChunk(
                    content=chunk_content,
                    is_complete=is_complete,
                    usage_info=usage_info,
                )
        except Exception as e:
            logger.error("Streaming generation failed", error=str(e))
            raise LLMException(f"Streaming generation failed: {e}")

    async def chat_stream(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[StreamingChunk]:
        """Chat with streaming responses"""
        try:
            azure_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    azure_messages.append(SystemMessage(msg["content"]))
                elif msg["role"] == "user":
                    azure_messages.append(UserMessage(msg["content"]))
                elif msg["role"] == "assistant":
                    azure_messages.append(AssistantMessage(msg["content"]))

            response = self.client.complete(
                stream=True,
                messages=azure_messages,
                model=self.model,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                top_p=kwargs.get("top_p", 1.0),
                # model_extras={"stream_options": {"include_usage": True}},
            )

            for update in response:
                chunk_content = ""
                usage_info = None
                is_complete = False

                if update.choices and update.choices[0].delta:
                    chunk_content = update.choices[0].delta.content or ""

                if update.usage:
                    usage_info = {
                        "completion_tokens": getattr(
                            update.usage, "completion_tokens", 0
                        ),
                        "prompt_tokens": getattr(update.usage, "prompt_tokens", 0),
                        "total_tokens": getattr(update.usage, "total_tokens", 0),
                    }
                    is_complete = True

                yield StreamingChunk(
                    content=chunk_content,
                    is_complete=is_complete,
                    usage_info=usage_info,
                )

        except Exception as e:
            logger.error("Streaming chat failed", error=str(e))
            raise LLMException(f"Streaming chat failed: {e}")
