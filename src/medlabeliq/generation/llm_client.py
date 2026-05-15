from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from medlabeliq.config.settings import settings


@lru_cache(maxsize=1)
def get_llm_client() -> OpenAI:
    if not settings.llm_api_key:
        raise RuntimeError(
            "LLM_API_KEY is empty. Add your provider API key to .env first."
        )

    return OpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


def generate_json_answer(
    *,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Call an OpenAI-compatible chat completion endpoint.
    """
    client = get_llm_client()

    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        seed=settings.llm_seed,
        max_completion_tokens=settings.llm_max_output_tokens,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("LLM returned an empty response.")

    return content