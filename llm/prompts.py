"""Prompt templates for LLM-powered query understanding."""

INTENT_CLASSIFIER_PROMPT = """\
You classify an AIC video search query into exactly one task type.

Task types:
- TEXTUAL_KIS: find one video moment/keyframe from a visual description.
- QA: find a video moment, then answer a question about the visual/OCR/ASR context.
- TRAKE: find an ordered sequence of events in one video.

Return only valid JSON with this shape:
{{
  "task_type": "TEXTUAL_KIS" | "QA" | "TRAKE",
  "confidence": number between 0 and 1,
  "reason": "short reason"
}}

Rules:
- If the query asks "what/how many/which/where/who" and needs an answer, use QA.
- If the query asks for multiple ordered events, use TRAKE.
- If the query only asks to find a described scene or moment, use TEXTUAL_KIS.
- Do not invent task types.

User query:
{query}
"""
