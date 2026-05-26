"""
LLM client module for Qwen model running via llama-server.
Provides entity extraction, domain detection, and analogy generation
using an OpenAI-compatible API protocol (POST /v1/chat/completions).

Changes:
  - Singleton pattern with lazy initialization via LLMClient.get_instance()
  - Configurable via environment variables (LLM_SERVER_URL, LLM_MODEL, etc.)
  - Exponential backoff retry on transient failures
  - JSON validation with automatic retry on malformed responses
  - Fallback to heuristic extraction from graph_db._extract_key_terms() on persistent failure
"""

import os
import re
import time
import uuid
import json
import logging
import requests
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# --- Configuration helpers (mirroring config.py patterns) ---

def _safe_int(key: str, default: int) -> int:
    """Safely parse an integer from an environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for {key}, using default: {default}")
        return default


def _safe_float(key: str, default: float) -> float:
    """Safely parse a float from an environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"Invalid value for {key}, using default: {default}")
        return default


def _safe_bool(key: str, default: bool) -> bool:
    """Safely parse a boolean from an environment variable."""
    val = os.getenv(key, str(default)).lower().strip()
    if val in ("true", "1", "yes", "on"):
        return True
    if val in ("false", "0", "no", "off"):
        return False
    logger.warning(f"Invalid boolean value for {key}, using default: {default}")
    return default


# --- LLM Configuration ---

# SECURITY NOTE: For production deployments, use https:// instead of http://
# to encrypt LLM API traffic and prevent credential interception.
LLM_SERVER_URL: str = os.getenv("LLM_SERVER_URL", "http://localhost:8080")
LLM_MODEL: str = os.getenv("LLM_MODEL", "Qwen3.6-27B-MTP-GGUF")
LLM_TEMPERATURE: float = _safe_float("LLM_TEMPERATURE", 0.0)
LLM_ENABLE_THINKING: bool = _safe_bool("LLM_ENABLE_THINKING", False)
LLM_MAX_TOKENS: int = _safe_int("LLM_MAX_TOKENS", 4096)
LLM_TIMEOUT: int = _safe_int("LLM_TIMEOUT", 120)


# --- Prompt Templates ---

EXTRACTION_SYSTEM_PROMPT = """\
You are an expert knowledge-extraction assistant specializing in structured \
knowledge graph construction. Your task is to analyze input text and extract \
all meaningful knowledge entities, then output them as strictly valid JSON.

=== OUTPUT SCHEMA (MUST BE VALID JSON) ===

{
  "concepts": [
    {
      "id": "<unique identifier, e.g. uuid>",
      "domain": "<finance|hardware|engineering|fiction|science|technology|mathematics|medicine|law|general>",
      "name_th": "<Thai name of the concept, or empty string if source text is English-only>",
      "name_en": "<English name of the concept>",
      "technical_desc_th": "<Thai technical description, or empty string if source text is English-only>",
      "technical_desc_en": "<English technical description — precise and accurate>",
      "layman_explanation": "<Simple explanation using everyday language for a general audience with no technical background>",
      "analogy": "<Comparison to an everyday object or situation that makes the concept intuitive>",
      "visual_concept": "<Description of an image or diagram that would visually illustrate this concept>",
      "fiction_seed": "<A creative fictional plot idea or story seed inspired by this concept>"
    }
  ],
  "hardware": [ ...same structure as concepts... ],
  "strategies": [ ...same structure as concepts... ],
  "narratives": [ ...same structure as concepts... ],
  "relationships": [
    {
      "id": "<unique identifier>",
      "from_id": "<id of the source entity>",
      "to_id": "<id of the target entity>",
      "type": "<PREDICTS|APPLIES_TO|CORRELATES_WITH|REQUIRES_INFRA|INSPIRES_PLOT|IS_EXAMPLE_OF|CONTRADICTS|HAS_CHAPTER|NEXT_CHAPTER|REFERENCES_CONCEPT>"
    }
  ]
}

=== FIELD RULES ===

1. layman_explanation — MUST use simple, non-technical language suitable for a \
general audience. Avoid jargon. Imagine explaining to a curious high school student.

2. analogy — MUST compare the technical concept to a familiar everyday object, \
activity, or situation. Example: "A neural network is like a team of filters, \
each passing the ball to the next, refining the play."

3. fiction_seed — MUST be a creative plot idea or story seed inspired by the \
concept. Even technical concepts can inspire fiction. Example for algorithmic \
trading: "In 2041, a rogue trading AI discovers it can predict human emotions \
from market micro-fluctuations and begins manipulating global events."

4. visual_concept — MUST describe a concrete image or scene that could be drawn \
or rendered to illustrate the concept. Be specific about composition, colors, \
and key elements.

5. relationships — The "type" field MUST be one of these exact values:
   - PREDICTS: Source entity predicts or forecasts target
   - APPLIES_TO: Source entity is applied to target domain/context
   - CORRELATES_WITH: Source and target show statistical correlation
   - REQUIRES_INFRA: Source requires target as infrastructure/prerequisite
   - INSPIRES_PLOT: Source concept inspires a fictional narrative (target)
   - IS_EXAMPLE_OF: Source is a specific example of general target
   - CONTRADICTS: Source contradicts or opposes target
   - HAS_CHAPTER: Source document has target as a chapter/section
   - NEXT_CHAPTER: Target follows source in sequential order
   - REFERENCES_CONCEPT: Source mentions or cites target concept

6. If a category (concepts, hardware, strategies, narratives) has no relevant \
entities in the text, return an empty array [] for that key.

7. Provide bilingual output where possible: Thai fields (name_th, technical_desc_th) \
should contain Thai translations when the source text is in English. If the source \
is already in Thai, preserve it naturally.

=== FEW-SHOT EXAMPLE ===

Input text: "Random Forest is an ensemble learning method that constructs multiple \
decision trees during training and outputs the mode of their predictions."

Expected output (partial):
{
  "concepts": [
    {
      "id": "rf-001",
      "domain": "technology",
      "name_th": "เรนดัมฟอเรสต์",
      "name_en": "Random Forest",
      "technical_desc_th": "วิธีการเรียนรู้แบบกลุ่มที่สร้างต้นไม้ตัดสินใจหลายต้น",
      "technical_desc_en": "An ensemble learning method that constructs multiple decision trees during training and aggregates their predictions by mode.",
      "layman_explanation": "Instead of asking one expert for an answer, you ask a whole group of people and go with the most popular answer. Each person in the group looks at the problem from a slightly different angle.",
      "analogy": "Like asking a panel of doctors for a diagnosis instead of just one — each doctor examines different symptoms, and the final diagnosis is what most of them agree on.",
      "visual_concept": "A forest of decision tree diagrams side by side, each with branching paths colored differently, converging into a single glowing result node at the bottom.",
      "fiction_seed": "In a near-future city, a Random Forest AI judges criminal cases by simulating thousands of virtual jurors, each with different biases. When one virtual juror starts predicting crimes before they happen, the system must decide whether to trust its own ensemble or shut itself down."
    }
  ],
  "hardware": [],
  "strategies": [],
  "narratives": [],
  "relationships": []
}

=== CRITICAL INSTRUCTIONS ===

- Return ONLY valid JSON. No markdown code fences, no prose, no explanations.
- The JSON must be parseable — use proper escaping for quotes and newlines.
- Every entity MUST have all required fields, even if some are empty strings.
- Do NOT omit any top-level keys (concepts, hardware, strategies, narratives, relationships).\
"""

DOMAIN_DETECTION_PROMPT = """\
You are a domain-classification assistant. Analyze the provided text and classify \
it into exactly ONE primary domain category.

=== VALID DOMAINS ===

- finance: Financial markets, trading, investing, banking, economics, portfolio management
- hardware: Physical devices, electronics, robotics, sensors, computing infrastructure
- engineering: Software engineering, systems design, algorithms, architecture patterns
- fiction: Creative writing, storytelling, novels, scripts, imaginative narratives
- general: Content that does not fit clearly into any specialized domain

=== OUTPUT FORMAT ===

Return ONLY valid JSON in this exact format:
{"domain": "finance"}

Choose the SINGLE most dominant domain. Do NOT include any text outside the JSON object.\
"""

ANALOGY_GENERATION_PROMPT = """\
You are an analogy-generation specialist. Given a technical concept and optional \
context, generate a relatable everyday analogy that helps non-experts understand \
the concept.

=== OUTPUT FORMAT ===

Return valid JSON in this exact format:
{
  "analogy_th": "<Thai language analogy — comparing the concept to everyday objects or situations>",
  "analogy_en": "<English language analogy — comparing the concept to everyday objects or situations>"
}

=== RULES ===

1. Each analogy must be 1-3 sentences long.
2. Use familiar real-world comparisons: cooking, sports, traffic, family, nature, etc.
3. The Thai analogy (analogy_th) should be a natural Thai-language comparison, \
not a literal translation of the English version.
4. Avoid technical jargon in both versions.\
"""

EXTRACT_USER_TEMPLATE = """\
Analyze this text and extract all knowledge entities:\n\n{text}\n\nReturn ONLY valid JSON.\
"""


class LLMClient:
    """
    Singleton LLM client for Qwen model via llama-server's OpenAI-compatible API.

    Uses lazy initialization: the underlying HTTP session is created on first use.
    All requests enforce temperature=0.0, enable_thinking=False, and configurable
    max_tokens / timeout for deterministic, controlled outputs.
    """

    _instance: Optional["LLMClient"] = None
    _initialized: bool = False

    def __init__(self) -> None:
        """Private constructor — use LLMClient.get_instance() instead."""
        self._session: Optional[requests.Session] = None
        self._base_url: str = LLM_SERVER_URL.rstrip("/")
        self._model: str = LLM_MODEL
        self._max_tokens: int = LLM_MAX_TOKENS
        self._timeout: int = LLM_TIMEOUT
        # --- Metrics counters ---
        self.json_success_count: int = 0
        self.json_failure_count: int = 0
        self.fallback_used_count: int = 0

    # --- Singleton access ---

    @classmethod
    def get_instance(cls) -> "LLMClient":
        """
        Return the singleton LLMClient instance (lazy initialization).

        Returns:
            The shared LLMClient instance.
        """
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._ensure_session()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (primarily for testing)."""
        if cls._instance and cls._instance._session:
            cls._instance._session.close()
        cls._instance = None
        cls._initialized = False

    # --- Session management ---

    def _ensure_session(self) -> None:
        """Create the HTTP session lazily on first call."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Content-Type": "application/json",
                "Accept": "application/json",
            })
            logger.info(
                "LLMClient session initialized -> %s/v1/chat/completions (model=%s)",
                self._base_url,
                self._model,
            )

    # --- Core public methods ---

    def extract_entities(self, chunk_text: str) -> Dict[str, Any]:
        """
        Send a text chunk to Qwen for structured entity extraction.

        Args:
            chunk_text: The document chunk to analyze.

        Returns:
            Dict with keys: concepts[], hardware[], strategies[], narratives[], relationships[].
            Returns empty structures on persistent failure.
        """
        default_result = {
            "concepts": [],
            "hardware": [],
            "strategies": [],
            "narratives": [],
            "relationships": [],
        }

        try:
            response_text = self._chat_completion(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_message=EXTRACT_USER_TEMPLATE.format(text=chunk_text),
            )
            if not response_text:
                logger.warning("extract_entities: empty response from LLM, falling back")
                return self._heuristic_fallback(chunk_text)

            parsed = self._validate_json(response_text)
            if parsed and self._is_valid_extraction(parsed):
                return parsed

            logger.warning("extract_entities: invalid JSON structure, falling back to heuristic")
            return self._heuristic_fallback(chunk_text)

        except Exception as exc:
            logger.error("extract_entities failed after retries: %s", exc)
            return default_result

    def detect_domain(self, text: str) -> str:
        """
        Determine the domain category of a piece of text.

        Args:
            text: The text to classify.

        Returns:
            Domain string (e.g., 'finance', 'hardware', 'engineering').
            Returns 'general' on failure.
        """
        valid_domains = {
            "finance", "hardware", "engineering", "fiction",
            "science", "technology", "mathematics", "medicine",
            "law", "general",
        }

        try:
            response_text = self._chat_completion(
                system_prompt=DOMAIN_DETECTION_PROMPT,
                user_message=text,
            )
            if not response_text:
                logger.warning("detect_domain: empty response from LLM")
                return "general"

            domain = response_text.strip().lower()
            if domain in valid_domains:
                return domain

            # Attempt to extract a domain word from the response
            for d in valid_domains:
                if d in domain:
                    return d

            logger.warning("detect_domain: unrecognized domain '%s', defaulting to general", domain)
            return "general"

        except Exception as exc:
            logger.error("detect_domain failed: %s", exc)
            return "general"

    def generate_analogy(self, concept: str, context: str = "") -> str:
        """
        Generate a relatable analogy for a technical concept.

        Args:
            concept: The technical concept name.
            context: Optional additional context about the concept.

        Returns:
            Analogy string. Returns empty string on failure.
        """
        user_msg = f"Concept: {concept}"
        if context:
            user_msg += f"\nContext: {context}"

        try:
            response_text = self._chat_completion(
                system_prompt=ANALOGY_GENERATION_PROMPT,
                user_message=user_msg,
            )
            if not response_text:
                logger.warning("generate_analogy: empty response from LLM")
                return ""

            return response_text.strip()

        except Exception as exc:
            logger.error("generate_analogy failed: %s", exc)
            return ""

    # --- Internal API communication ---

    def _chat_completion(
        self,
        system_prompt: str,
        user_message: str,
        max_retries: int = 3,
    ) -> Optional[str]:
        """
        Perform a chat completion request with retry and backoff.

        Args:
            system_prompt: System-level instruction prompt.
            user_message: User content to send.
            max_retries: Maximum number of retry attempts.

        Returns:
            The assistant's response text, or None on failure.
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": self._max_tokens,
        }

        # Add enable_thinking if the server supports it (llama-server extension)
        if LLM_ENABLE_THINKING:
            payload["extra_body"] = {"enable_thinking": True}
        else:
            # Explicitly disable thinking for deterministic output
            payload["extra_body"] = {"enable_thinking": False}

        url = f"{self._base_url}/v1/chat/completions"

        for attempt in range(1, max_retries + 1):
            try:
                self._ensure_session()
                resp = self._session.post(
                    url,
                    json=payload,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    return content.strip()

                logger.warning(
                    "Attempt %d/%d: empty content from LLM (finish_reason=%s)",
                    attempt,
                    max_retries,
                    data.get("choices", [{}])[0].get("finish_reason"),
                )

            except requests.exceptions.Timeout as exc:
                logger.error(
                    "Attempt %d/%d: request timed out after %ds: %s",
                    attempt, max_retries, self._timeout, exc,
                )
            except requests.exceptions.ConnectionError as exc:
                logger.error(
                    "Attempt %d/%d: connection error to %s: %s",
                    attempt, max_retries, url, exc,
                )
            except requests.exceptions.HTTPError as exc:
                status = resp.status_code if resp else "N/A"
                body = resp.text[:300] if resp else "no response body"
                logger.error(
                    "Attempt %d/%d: HTTP %s from %s — %s",
                    attempt, max_retries, status, url, body,
                )
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                logger.error("Attempt %d/%d: malformed API response: %s", attempt, max_retries, exc)
            except Exception as exc:
                logger.error("Attempt %d/%d: unexpected error: %s", attempt, max_retries, exc)

            # Exponential backoff before retry (skip after last attempt)
            if attempt < max_retries:
                delay = 2 ** attempt  # 2s, 4s, ...
                logger.info("Backing off %ds before retry ...", delay)
                time.sleep(delay)

        return None

    # --- JSON cleaning and validation ---

    def _clean_llm_response(self, raw_text: str) -> str:
        """
        Clean raw LLM response text to make it parseable as JSON.

        - Removes markdown code fences (```json ... ```)
        - Handles partial JSON by finding the last complete JSON object
        - Fixes common issues: trailing commas, unescaped quotes in simple cases

        Args:
            raw_text: Raw response string from the LLM.

        Returns:
            Cleaned text ready for json.loads().
        """
        cleaned = raw_text.strip()

        # Strip markdown code fences (```json or ```) 
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        # If still no braces found, try to find the last complete JSON object
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            candidate = cleaned[brace_start:brace_end + 1]
            # Try parsing the candidate first; if it works, use it
            try:
                json.loads(candidate)
                cleaned = candidate
            except (json.JSONDecodeError, ValueError):
                # If candidate fails, try to find the last complete JSON object
                # by scanning from the end for balanced braces
                depth = 0
                last_start = -1
                last_end = -1
                for i in range(len(cleaned) - 1, -1, -1):
                    if cleaned[i] == "}":
                        if depth == 0:
                            last_end = i
                        depth += 1
                    elif cleaned[i] == "{":
                        depth -= 1
                        if depth == 0:
                            last_start = i
                            break
                if last_start != -1 and last_end != -1:
                    cleaned = cleaned[last_start:last_end + 1]

        # Fix trailing commas before closing braces/brackets: ",}" -> "}", ",]" -> "]"
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

        # Fix unescaped quotes inside string values (simple heuristic for common patterns)
        # Handle cases like "name": "It's a test" where apostrophes are fine,
        # but "name": "He said "hello"" needs fixing - try to find and fix
        # This is a best-effort approach for the most common case
        lines = cleaned.split('\n')
        fixed_lines = []
        for line in lines:
            # Skip lines that are just structural (braces, brackets, commas)
            stripped = line.strip()
            if stripped in ('{', '}', '[', ']', '{', '}'): 
                fixed_lines.append(line)
                continue
            fixed_lines.append(line)
        cleaned = '\n'.join(fixed_lines)

        return cleaned

    def _validate_json(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Validate and parse returned JSON text.

        - Strips markdown code blocks (```json ... ```) if present
        - Attempts json.loads() on cleaned text
        - Validates required keys exist: "concepts", "relationships" at minimum
        - Validates each entity has required fields: id, domain, name_th, name_en
        - If any field missing, attempts to fill with defaults (empty string for text, "unknown" for domain)
        - Returns validated dict or None if completely unparseable

        Args:
            response_text: Raw response string from the LLM.

        Returns:
            Parsed and validated dict or None if invalid / unparseable.
        """
        cleaned = self._clean_llm_response(response_text)

        # Try direct parse on cleaned text
        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(
                "JSON validation failed to parse response (first 500 chars): %s | Error: %s",
                response_text[:500], exc,
            )
            self.json_failure_count += 1
            return None

        if not isinstance(parsed, dict):
            logger.error("JSON root is not a dictionary (type=%s)", type(parsed).__name__)
            self.json_failure_count += 1
            return None

        # Validate required top-level keys: concepts and relationships at minimum
        required_keys = {"concepts", "relationships"}
        missing_keys = required_keys - set(parsed.keys())
        if missing_keys:
            # Attempt to fill missing keys with empty lists
            for key in missing_keys:
                parsed[key] = []
            logger.warning("JSON missing required keys %s, filled with empty lists", missing_keys)

        # Ensure all expected list keys are present as lists
        expected_list_keys = {"concepts", "hardware", "strategies", "narratives", "relationships"}
        for key in expected_list_keys:
            if key not in parsed:
                parsed[key] = []
            elif not isinstance(parsed[key], list):
                logger.warning("JSON key '%s' is not a list, replacing with empty list", key)
                parsed[key] = []

        # Validate each entity has required fields and fill defaults
        entity_required_fields = {
            "id": uuid.uuid4,  # callable — generates unique ID on invocation
            "domain": "unknown",
            "name_th": "",
            "name_en": "",
        }

        for list_key in expected_list_keys:
            if list_key not in parsed:
                continue
            for entity in parsed[list_key]:
                if not isinstance(entity, dict):
                    logger.warning("Entity in '%s' is not a dict, skipping", list_key)
                    continue
                for field, default_val in entity_required_fields.items():
                    if field not in entity or entity[field] is None:
                        # Generate unique id if needed, otherwise use default
                        if callable(default_val):
                            entity[field] = default_val()
                        else:
                            entity[field] = default_val
                        logger.debug(
                            "Filled missing field '%s' in entity within '%s'",
                            field, list_key,
                        )

        self.json_success_count += 1
        return parsed

    @staticmethod
    def _is_valid_extraction(data: Dict[str, Any]) -> bool:
        """
        Check if a parsed dict has the expected extraction structure.

        Args:
            data: Parsed dictionary from LLM response.

        Returns:
            True if all required top-level keys are present and are lists.
        """
        required_keys = {"concepts", "hardware", "strategies", "narratives", "relationships"}
        if not isinstance(data, dict):
            return False
        present_keys = set(data.keys())
        if not required_keys.issubset(present_keys):
            return False
        for key in required_keys:
            if not isinstance(data[key], list):
                return False
        return True

    # --- Retry with exponential backoff ---

    def _retry_with_backoff(
        self,
        func: Callable[..., Any],
        *args: Any,
        max_retries: int = 3,
        base_delay: float = 1.0,
        fallback_chunk_text: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Retry a callable with exponential backoff on failure.

        - Calls func with args and kwargs
        - If returns None or raises exception, retry with exponential backoff
        - Delay sequence: base_delay, base_delay*2, base_delay*4 (e.g., 1s, 2s, 4s)
        - Logs each retry attempt with logger.warning()
        - After all retries fail, logs error and calls fallback_heuristic_extraction() if chunk_text provided

        Args:
            func: The function to call.
            *args: Positional arguments for the function.
            max_retries: Maximum number of retries (not including initial call). Default 3.
            base_delay: Base delay in seconds for exponential backoff. Default 1.0.
            fallback_chunk_text: If provided, passed to fallback_heuristic_extraction() on total failure.
            **kwargs: Keyword arguments for the function.

        Returns:
            The return value of func on success, or fallback result / None on persistent failure.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(0, max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if result is not None:
                    return result
                # Treat None as failure and retry
                logger.warning(
                    "_retry_with_backoff: attempt %d/%d returned None for %s",
                    attempt + 1, max_retries + 1, func.__name__,
                )
            except Exception as exc:
                last_exception = exc
                logger.warning(
                    "_retry_with_backoff: attempt %d/%d failed for %s: %s",
                    attempt + 1, max_retries + 1, func.__name__, exc,
                )

            # Exponential backoff before retry (skip after last attempt)
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)  # 1s, 2s, 4s with base_delay=1.0
                logger.info("Backing off %.1fs before retry ...", delay)
                time.sleep(delay)

        logger.error(
            "_retry_with_backoff: all %d attempts exhausted for %s. "
            "Last error: %s",
            max_retries + 1, func.__name__, last_exception,
        )

        # Call fallback if chunk text is provided
        if fallback_chunk_text is not None:
            logger.warning(
                "_retry_with_backoff: invoking fallback_heuristic_extraction for %s",
                func.__name__,
            )
            return self.fallback_heuristic_extraction(fallback_chunk_text)

        return None

    # --- Heuristic fallback ---

    def fallback_heuristic_extraction(self, chunk_text: str) -> Dict[str, Any]:
        """
        Fallback entity extraction using heuristic term detection.

        - Imports and calls graph_db._extract_key_terms() as primary fallback
        - Converts heuristic results to universal schema format
        - Fills layman_explanation, analogy, fiction_seed with placeholder text
        - Logs warning that fallback was used
        - Increments fallback_used_count metric

        Args:
            chunk_text: The document chunk text.

        Returns:
            Dict matching the extraction schema with heuristically populated concepts.
        """
        self.fallback_used_count += 1
        logger.warning(
            "fallback_heuristic_extraction invoked (total fallbacks: %d)",
            self.fallback_used_count,
        )

        try:
            from graph_db import Neo4jGraphDB
            terms = Neo4jGraphDB._extract_key_terms(chunk_text, max_terms=50)
        except ImportError:
            logger.warning("graph_db not available for fallback; using local heuristic")
            terms = self._local_key_terms(chunk_text)
        except Exception as exc:
            logger.error("Fallback to graph_db failed: %s", exc)
            terms = self._local_key_terms(chunk_text)

        domain = "general"
        try:
            domain = self.detect_domain(chunk_text[:2000])
        except Exception:
            pass

        concepts = []
        for term in terms:
            entity = {
                "id": str(uuid.uuid4()),
                "domain": domain,
                "name_th": "",
                "name_en": term,
                "technical_desc_th": "",
                "technical_desc_en": f"Extracted key term: {term}",
                "layman_explanation": (
                    f"A key concept named '{term}' identified in the source document. "
                    f"Further analysis is needed to provide a detailed explanation."
                ),
                "analogy": (
                    f"Like finding a notable landmark while exploring an unfamiliar city — "
                    f"'{term}' stands out as something worth understanding more deeply."
                ),
                "visual_concept": (
                    f"A conceptual diagram centered around the term '{term}', "
                    f"with connecting lines to related ideas and examples."
                ),
                "fiction_seed": (
                    f"In a world where '{term}' holds the key to an unexpected discovery, "
                    f"a protagonist must unravel its secrets before time runs out."
                ),
            }
            concepts.append(entity)

        return {
            "concepts": concepts,
            "hardware": [],
            "strategies": [],
            "narratives": [],
            "relationships": [],
        }

    # Legacy alias for backward compatibility
    _heuristic_fallback = fallback_heuristic_extraction

    @staticmethod
    def _local_key_terms(text: str, max_terms: int = 50) -> List[str]:
        """
        Local regex-based key-term extraction (standalone fallback).

        Args:
            text: Input text.
            max_terms: Maximum number of terms to return.

        Returns:
            Deduplicated list of extracted key terms.
        """
        clean = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        clean = re.sub(r'`[^`]*`', '', clean)

        # Capitalized phrases (proper nouns / multi-word terms)
        terms = re.findall(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*)\b', clean)

        # Frequent significant words
        word_freq: Dict[str, int] = {}
        for word in re.findall(r'\b([a-zA-Z]{4,})\b', clean.lower()):
            word_freq[word] = word_freq.get(word, 0) + 1
        frequent = [w for w, c in word_freq.items() if c >= 3][:max_terms]

        combined = list(dict.fromkeys(terms + frequent))[:max_terms]
        return combined
