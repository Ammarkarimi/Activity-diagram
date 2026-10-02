from __future__ import annotations

import os
import time
from typing import TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from src.llm.usage import UsageTracker

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class OpenAIClient:

    def __init__(self, model: str | None = None) -> None:

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set."
            )

        self.model = model or os.getenv(
            "OPENAI_MODEL",
            "gpt-5.4-mini"
        )

        self.client = OpenAI(
            api_key=api_key,
            timeout=float(
                os.getenv(
                    "OPENAI_TIMEOUT",
                    "120"
                )
            )
        )

        # Token usage of every call (observation only, see src/llm/usage.py).
        self.usage = UsageTracker()

        self.max_retries = int(
            os.getenv(
                "MAX_RETRIES",
                "3"
            )
        )

    def complete(
        self,
        *,
        system: str,
        user: str,
        response_model: type[T],
    ) -> T:

        last_error = None

        for attempt in range(self.max_retries):

            try:

                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {
                            "role": "system",
                            "content": system,
                        },
                        {
                            "role": "user",
                            "content": user,
                        },
                    ],
                    text_format=response_model,
                )

                self.usage.record(
                    self.model,
                    response_model.__name__,
                    getattr(response, "usage", None),
                    failed=response.output_parsed is None,
                )

                if response.output_parsed is None:
                    raise RuntimeError(
                        "OpenAI returned no structured output."
                    )

                return response.output_parsed

            except Exception as exc:

                last_error = exc

                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"OpenAI request failed after retries: {last_error}"
        )