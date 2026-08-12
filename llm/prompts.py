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

QUERY_PLAN_PARSER_PROMPT = """\
You parse an AIC video search query into exactly one QueryPlan JSON object.

Task types:
- TEXTUAL_KIS: find one video moment/keyframe from a visual description.
- QA: find a video moment, then answer a question about visual/OCR/ASR context.
- TRAKE: find an ordered sequence of events in one video.

Return only valid JSON with this shape:
{{
  "task_type": "TEXTUAL_KIS" | "QA" | "TRAKE",
  "search_description": "normalized visual search text",
  "question": "question text for QA, otherwise null",
  "events": [
    {{"event_id": 1, "description": "first ordered event"}}
  ],
  "entities": [
    {{
      "name": "person/object/location name",
      "type": "person" | "object" | "vehicle" | "location" | "unknown",
      "attributes": ["short visual attributes"],
      "actions": [
        {{"verb": "action phrase", "target": "optional target or null"}}
      ]
    }}
  ],
  "objects": ["visible objects or people"],
  "actions": ["action phrases"],
  "scene": ["locations, setting, time of day, environment"],
  "positive_constraints": ["must-match visual/text constraints"],
  "negative_constraints": ["must-not-match constraints"],
  "metadata_keywords": ["OCR/ASR/logo/sign/license plate keywords"],
  "confidence": number between 0 and 1
}}

Rules:
- Return JSON only. Do not wrap it in markdown.
- The task_type has already been classified. Use that task_type exactly.
- Do not override task_type. Only extract QueryPlan fields for the classified task type.
- Keep event_id values 1-based and ordered for TRAKE.
- For QA, keep the question in "question" and use "search_description" for the visual evidence needed to answer it.
- Use Vietnamese without accents if the query is Vietnamese without accents.
- Do not invent details that are not implied by the user query.

Final classified intent:
{intent}

User query:
{query}
"""
