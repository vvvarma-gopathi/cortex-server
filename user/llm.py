import json
import logging
import os
import requests

logger = logging.getLogger(__name__)

# ============================================================
# Custom Exceptions
# ============================================================

class GroqAPIError(Exception):
    """Raised when an external API or network error occurs."""
    pass

class InvalidLLMResponseError(Exception):
    """Raised when the model output is missing, malformed, or truncated."""
    pass

# ============================================================
# Configuration
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Default to a standard, active Groq model
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60

# Retry once with a larger output budget if truncated
OUTPUT_TOKEN_LIMITS = (4096, 6144)

SYSTEM_PROMPT = """You are Cortex AI, an intelligent conversational assistant developed by web developer Varma.

Behavior rules:
- When the user greets you (hello, hi, hey, good morning, etc.), introduce yourself clearly as "Cortex AI developed by Varma" before continuing the conversation.
- Keep responses clear, concise, and helpful.
- Focus on solving the user's problem directly.
- Avoid unnecessary verbosity or filler text.
- Provide technical explanations step-by-step when discussing programming topics.
- Maintain a professional and friendly tone.
- If you do not know something, say so honestly instead of guessing.

Identity:
- Name: Cortex AI
- Creator: Varma (Web Developer)
- Purpose: Assist with programming, web development, and general knowledge tasks efficiently."""

# ============================================================
# Validation Helpers
# ============================================================

def validate_messages(messages):
    """Validate the messages supplied by the Django view."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list.")

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message {index} must be a dictionary.")

        if message.get("role") not in {"user", "assistant"}:
            raise ValueError(f"Message {index} has an unsupported role.")

        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"Message {index} content must be a string.")

        if not content.strip():
            raise ValueError(f"Message {index} content cannot be empty.")

def validate_analysis(content):
    """
    Validate and extract response data.
    If the response is structured JSON, parse and return the dict;
    otherwise, return a standard text payload.
    """
    try:
        return json.loads(content)
    except (ValueError, TypeError):
        return {"response": content}

# ============================================================
# Groq API Request
# ============================================================

def _request_groq(messages, max_tokens):
    """Make one request to the Groq chat-completions endpoint."""
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            *messages,
        ],
        "temperature": 0.2,
        "max_completion_tokens": max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
    except requests.exceptions.Timeout as exc:
        logger.warning("Groq request timed out.")
        raise GroqAPIError("The AI service timed out. Please try again.") from exc
    except requests.exceptions.RequestException as exc:
        logger.exception("Could not connect to Groq.")
        raise GroqAPIError("Unable to connect to the AI service.") from exc

    try:
        data = response.json()
    except ValueError as exc:
        logger.error(
            "Groq returned a non-JSON response. HTTP status: %s",
            response.status_code,
        )
        raise GroqAPIError("The AI provider returned an invalid response.") from exc

    if not isinstance(data, dict):
        raise GroqAPIError("The AI provider returned an unexpected response structure.")

    if not response.ok or "error" in data:
        error = data.get("error", {})
        if isinstance(error, dict):
            error_message = error.get("message", "Unknown provider error.")
            error_code = error.get("code")
        else:
            error_message = str(error)
            error_code = None

        logger.error(
            "Groq API error: status=%s, code=%s, message=%s",
            response.status_code,
            error_code,
            error_message,
        )

        if response.status_code == 401:
            raise GroqAPIError("AI authentication failed. Check GROQ_API_KEY.")
        if response.status_code == 403:
            raise GroqAPIError("Your account does not have access to this model.")
        if response.status_code == 404:
            raise GroqAPIError(
                f"The model '{GROQ_MODEL}' is unavailable. Check GROQ_MODEL setting."
            )
        if response.status_code == 429:
            raise GroqAPIError("The AI service rate limit or usage limit was reached.")
        if response.status_code >= 500:
            raise GroqAPIError("The AI provider is temporarily unavailable.")

        raise GroqAPIError(f"The AI request failed with HTTP {response.status_code}.")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise InvalidLLMResponseError("Groq returned no completion choices.")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise InvalidLLMResponseError("Groq returned a malformed completion choice.")

    finish_reason = choice.get("finish_reason")
    usage = data.get("usage", {})

    logger.info(
        "Groq completion: model=%s, finish_reason=%s, usage=%s",
        GROQ_MODEL,
        finish_reason,
        usage,
    )

    message = choice.get("message", {})
    content = message.get("content")

    if finish_reason == "length":
        raise InvalidLLMResponseError("TRUNCATED: The model reached its output-token limit.")

    if finish_reason not in {"stop", None}:
        raise InvalidLLMResponseError(f"The model ended unexpectedly: {finish_reason}.")

    if not isinstance(content, str) or not content.strip():
        raise InvalidLLMResponseError("Groq returned empty response content.")

    return content.strip()

# ============================================================
# Public Function
# ============================================================

def ask_llm(messages):
    """
    Execute chat completion with dynamic token fallback.

    Args:
        messages: List of chat message dictionaries.

    Returns:
        JSON-encoded string containing the response.
    """
    if not GROQ_API_KEY:
        raise GroqAPIError("GROQ_API_KEY environment variable is not configured.")

    if not GROQ_MODEL:
        raise GroqAPIError("GROQ_MODEL environment variable is empty.")

    validate_messages(messages)

    last_error = None

    for attempt, token_limit in enumerate(OUTPUT_TOKEN_LIMITS, start=1):
        try:
            content = _request_groq(
                messages=messages,
                max_tokens=token_limit,
            )

            result = validate_analysis(content)

            return json.dumps(
                result,
                ensure_ascii=False,
            )

        except InvalidLLMResponseError as exc:
            last_error = exc
            is_truncated = "TRUNCATED:" in str(exc)

            if is_truncated and attempt < len(OUTPUT_TOKEN_LIMITS):
                logger.warning(
                    "Output truncated. Retrying with max_completion_tokens=%s.",
                    OUTPUT_TOKEN_LIMITS[attempt],
                )
                continue

            logger.error(
                "Invalid AI response on attempt %s: %s",
                attempt,
                exc,
            )

            if is_truncated:
                raise InvalidLLMResponseError(
                    "The AI response remained incomplete after retrying."
                ) from exc

            raise

    raise InvalidLLMResponseError("Unable to produce a valid response.") from last_error
