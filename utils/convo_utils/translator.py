"""
translator.py — Text and question translation via GPT-4o-mini.

Adapted from convo-backend-v2 QuestionTranslator (proprietary, 2024).
Added translate_text() for translating arbitrary transcript chunks.

Usage
-----
    import asyncio
    from convo_utils.translator import QuestionTranslator

    translator = QuestionTranslator()

    # Translate a transcript chunk (for translate-first RAG pipeline)
    english = asyncio.run(translator.translate_text(
        "Ik vind het erg moeilijk om met mijn collega's samen te werken.",
        from_lang="nl", to_lang="en"
    ))
    # → "I find it very difficult to collaborate with my colleagues."

    # Translate a list of question dicts
    translated_questions = asyncio.run(translator.translate_questions(
        questions=[{"id": "q1", "question": "Hoe bevalt uw werk?"}],
        from_lang="nl", to_lang="en"
    ))

Thesis use
----------
- Implement the translate-first RAG pipeline (ablation Group D):
    1. Translate Dutch transcript chunks → English
    2. Query the model in English
    3. Translate answer back → Dutch
  This eliminates the language gap without touching the model weights.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("convo-utils-translator")

# Full language code → name map
_LANG_NAMES: Dict[str, str] = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
    "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "ar": "Arabic", "hi": "Hindi", "sv": "Swedish", "no": "Norwegian",
    "da": "Danish", "fi": "Finnish", "yue": "Cantonese", "zh-hk": "Cantonese",
}


class QuestionTranslator:
    """
    Translates questions and arbitrary text using GPT-4o-mini.
    Includes in-memory caching to avoid duplicate API calls.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. Falls back to OPENAI_API_KEY environment variable.
        """
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self._cache: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # New method: translate arbitrary text (for transcript chunks)
    # ------------------------------------------------------------------

    async def translate_text(self, text: str, from_lang: str, to_lang: str) -> str:
        """
        Translate a single piece of text (transcript chunk, answer, etc.).

        Parameters
        ----------
        text : str
            Text to translate.
        from_lang : str
            Source language code (e.g. 'nl', 'en').
        to_lang : str
            Target language code.

        Returns
        -------
        str
            Translated text. Returns original text on failure.
        """
        from_lang = (from_lang or "en").lower()[:2]
        to_lang = (to_lang or "en").lower()[:2]

        if from_lang == to_lang:
            return text

        cache_key = f"text_{from_lang}_{to_lang}_{hash(text)}"
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore

        from_name = _LANG_NAMES.get(from_lang, from_lang.upper())
        to_name = _LANG_NAMES.get(to_lang, to_lang.upper())

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Translate the following text from {from_name} to {to_name}. "
                            "Preserve meaning, tone, and any speaker labels exactly. "
                            "Output ONLY the translated text, nothing else."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            result = response.choices[0].message.content or text
            result = result.strip()
            self._cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"[Translator] translate_text failed ({e}), returning original")
            return text

    # ------------------------------------------------------------------
    # Original method: translate question dicts
    # ------------------------------------------------------------------

    async def translate_questions(
        self,
        questions: List[Dict],
        from_lang: str,
        to_lang: str,
    ) -> List[Dict]:
        """
        Translate a list of question dictionaries.

        Parameters
        ----------
        questions : List[dict]
            Each dict must have a 'question' key.
        from_lang : str
            Source language code.
        to_lang : str
            Target language code.

        Returns
        -------
        List[dict]
            Questions with translated 'question' field.
            Adds: 'original_question', 'translated_from', 'translated_to'.
        """
        from_lang = (from_lang or "en").lower()[:2]
        to_lang = (to_lang or "en").lower()[:2]

        if from_lang == to_lang:
            return questions

        cache_key = (
            f"questions_{from_lang}_{to_lang}_"
            + "_".join([q.get("id", str(i)) for i, q in enumerate(questions)])
        )
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore

        question_texts = [q.get("question", "") for q in questions]
        prompt = self._build_batch_prompt(question_texts, from_lang, to_lang)

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional translator. Translate the questions "
                            "accurately while preserving their intent and tone. "
                            "Output ONLY the translations, one per line."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            translated_text = response.choices[0].message.content or ""
            translated_lines = [
                line.strip() for line in translated_text.split("\n") if line.strip()
            ]

            translated_questions = []
            for i, question in enumerate(questions):
                tq = question.copy()
                if i < len(translated_lines):
                    tq["question"] = translated_lines[i]
                    tq["original_question"] = question["question"]
                    tq["translated_from"] = from_lang
                    tq["translated_to"] = to_lang
                translated_questions.append(tq)

            self._cache[cache_key] = translated_questions
            logger.info(f"[Translator] Translated {len(translated_questions)} questions {from_lang}→{to_lang}")
            return translated_questions

        except Exception as e:
            logger.error(f"[Translator] translate_questions failed ({e}), returning originals")
            return questions

    def _build_batch_prompt(self, questions: List[str], from_lang: str, to_lang: str) -> str:
        from_name = _LANG_NAMES.get(from_lang, from_lang.upper())
        to_name = _LANG_NAMES.get(to_lang, to_lang.upper())
        prompt = (
            f"Translate these interview questions from {from_name} to {to_name}.\n"
            "Maintain the professional tone and exact meaning.\n"
            "Output ONLY the translated questions, one per line, no numbering or bullets.\n\n"
            "Questions to translate:\n"
        )
        for q in questions:
            prompt += f"{q}\n"
        return prompt


# ---------------------------------------------------------------------------
# Singleton helper
# ---------------------------------------------------------------------------

_translator: Optional[QuestionTranslator] = None
_translator_lock = asyncio.Lock()


async def get_translator() -> QuestionTranslator:
    """Get or create the singleton translator instance."""
    global _translator
    if _translator is None:
        async with _translator_lock:
            if _translator is None:
                _translator = QuestionTranslator()
    return _translator
