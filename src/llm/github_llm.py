from typing import Any, AsyncIterator, Dict, List, Optional

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential

from src.config.settings import settings
from src.core.exceptions import LLMException
from src.core.interfaces import LLMInterface, StreamingChunk
from src.utils.logging import logger


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

            # Check if model requires max_completion_tokens instead of max_tokens
            # Newer OpenAI models (gpt-5, o1, etc.) use max_completion_tokens
            self._use_completion_tokens = self._requires_completion_tokens(self.model)

            # Track whether this model supports custom temperature values.
            # Some models (o1, o3, o4, gpt-4.1, …) only accept the default (1).
            # We start optimistic and disable on the first unsupported_value error.
            self._use_temperature = self._model_supports_temperature(self.model)
            if not self._use_temperature:
                logger.info(
                    "Custom temperature disabled for model (only default=1 supported)",
                    model=self.model,
                )

        except Exception as e:
            logger.error("GitHub LLM initialization failed", error=str(e))
            raise LLMException(f"Failed to initialize GitHub LLM: {e}")

    def _requires_completion_tokens(self, model: str) -> bool:
        """Check if model requires max_completion_tokens instead of max_tokens"""
        # Last verified against GitHub Models catalogue — update if new reasoning
        # models are added that use max_completion_tokens.
        models_requiring_completion_tokens = [
            "gpt-5",
            "o1-",
            "o3-",
            "o4-",  # OpenAI reasoning models
        ]
        model_lower = model.lower()
        return any(m in model_lower for m in models_requiring_completion_tokens)

    def _model_supports_temperature(self, model: str) -> bool:
        """Return False for models that only accept the default temperature (1)."""
        # Last verified against GitHub Models catalogue — update if new models
        # are added that reject custom temperature values.
        # Reasoning / constrained models on GitHub Models that reject custom temperature
        no_temp_families = ["o1", "o3", "o4", "gpt-4.1"]
        # Compare against the base model name (strip org prefix, e.g. "openai/o3-mini" → "o3-mini")
        base = model.lower().split("/")[-1]
        return not any(base.startswith(f) for f in no_temp_families)

    @staticmethod
    def _is_temperature_error(exc: Exception) -> bool:
        """Return True when the exception is an unsupported temperature value error."""
        msg = str(exc).lower()
        return "unsupported_value" in msg and "temperature" in msg

    def _get_max_tokens_param(self, max_tokens_value: int) -> dict:
        """Get the appropriate max tokens parameter for the model"""
        if self._use_completion_tokens:
            # Azure AI Inference SDK doesn't support max_completion_tokens
            # For newer models, we omit the parameter and let the model use its default
            return {}
        return {"max_tokens": max_tokens_value}

    def _build_params(
        self,
        messages: list,
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
    ) -> dict:
        """Build parameters dict, omitting temperature when the model doesn't support it."""
        params: dict = {
            "stream": True,
            "messages": messages,
            "model": self.model,
            "top_p": top_p,
        }
        if self._use_temperature:
            params["temperature"] = temperature
        params.update(self._get_max_tokens_param(max_tokens))
        return params

    async def generate_stream(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[StreamingChunk]:
        """Generate text from prompt with streaming.

        Automatically retries without the ``temperature`` parameter when the
        model signals it is unsupported (GitHub Models ``unsupported_value``
        error), then remembers this for all future calls on this instance.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens_val = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", 1.0)
        messages = [UserMessage(prompt)]

        for attempt in range(2):
            chunks_yielded = 0
            try:
                params = self._build_params(
                    messages, temperature, max_tokens_val, top_p
                )
                response = self.client.complete(**params)  # type: ignore[attr-defined]

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

                    chunks_yielded += 1
                    yield StreamingChunk(
                        content=chunk_content,
                        is_complete=is_complete,
                        usage_info=usage_info,
                    )
                break  # stream completed successfully

            except Exception as e:
                # Only retry (once) if the error is about unsupported temperature
                # AND we haven't emitted any content yet (safe to restart).
                if (
                    attempt == 0
                    and chunks_yielded == 0
                    and self._is_temperature_error(e)
                ):
                    logger.warning(
                        "Model does not support custom temperature — retrying without it",
                        model=self.model,
                        temperature=temperature,
                    )
                    self._use_temperature = False
                    continue  # retry without temperature in params
                logger.error("Streaming generation failed", error=str(e))
                raise LLMException(f"Streaming generation failed: {e}") from e

    async def chat_stream(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[StreamingChunk]:
        """Chat with streaming responses.

        Same auto-retry logic as ``generate_stream`` for unsupported temperature.
        """
        azure_messages = []
        for msg in messages:
            if msg["role"] == "system":
                azure_messages.append(SystemMessage(msg["content"]))
            elif msg["role"] == "user":
                azure_messages.append(UserMessage(msg["content"]))
            elif msg["role"] == "assistant":
                azure_messages.append(AssistantMessage(msg["content"]))

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens_val = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", 1.0)

        for attempt in range(2):
            chunks_yielded = 0
            try:
                params = self._build_params(
                    azure_messages, temperature, max_tokens_val, top_p
                )
                response = self.client.complete(**params)  # type: ignore[attr-defined]

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

                    chunks_yielded += 1
                    yield StreamingChunk(
                        content=chunk_content,
                        is_complete=is_complete,
                        usage_info=usage_info,
                    )
                break  # stream completed successfully

            except Exception as e:
                if (
                    attempt == 0
                    and chunks_yielded == 0
                    and self._is_temperature_error(e)
                ):
                    logger.warning(
                        "Model does not support custom temperature — retrying without it",
                        model=self.model,
                        temperature=temperature,
                    )
                    self._use_temperature = False
                    continue  # retry without temperature in params
                logger.error("Streaming chat failed", error=str(e))
                raise LLMException(f"Streaming chat failed: {e}") from e
