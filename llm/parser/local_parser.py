"""Local query parser for Day 3.

This module turns a raw user query into the Day 1 ``QueryPlan`` contract. The
implementation is intentionally heuristic so the pipeline can run without an
API while keeping the same output shape a future LLM parser must produce.
"""

from __future__ import annotations

import re
from typing import Iterable

from ..classifier import classify_intent, normalize_text
from ..schemas import Entity, EntityAction, IntentClassification, QueryPlan
from ..task_types import TaskType
from ..validator import validate_query_plan

SEARCH_PREFIXES = (
    "tim chuoi su kien",
    "tim chuoi",
    "tim doan",
    "tim khoanh khac",
    "tim hinh anh",
    "tim khung hinh",
    "tim canh",
    "tim",
)

QA_CONTEXT_PREFIXES = (
    "trong canh",
    "trong khung hinh",
    "trong video",
)

TEMPORAL_SPLIT_PATTERN = re.compile(
    r"\s*(?:,|\broi\b|\bsau do\b|\btiep theo\b|\btruoc khi\b|\bsau khi\b|\bva\b)\s*"
)

OBJECT_KEYWORDS = (
    "nguoi dan ong",
    "nguoi phu nu",
    "nguoi cong nhan",
    "nguoi ban hang",
    "nguoi giao hang",
    "nhan vien",
    "hanh khach",
    "em be",
    "dua tre",
    "cau thu",
    "dau bep",
    "khach",
    "nguoi",
    "xe buyt",
    "xe may",
    "xe tai",
    "xe",
    "cua xe",
    "cop xe",
    "bien so",
    "bien hieu",
    "logo",
    "chu",
    "dien thoai",
    "laptop",
    "may tinh",
    "man hinh",
    "bong bay",
    "bong",
    "vali",
    "goi hang",
    "tui do",
    "chai nuoc",
    "banh",
    "sach",
    "dia thuc an",
    "ban",
    "ghe",
    "cua hang",
    "toa nha",
    "tu lanh",
    "tu",
    "noi",
    "rau",
    "mon an",
    "micro",
    "o",
    "cho",
    "con vat",
    "tau hoa",
    "thuyen",
)

SCENE_KEYWORDS = (
    "cong vien",
    "nga tu",
    "buoi toi",
    "tram xe",
    "quan ca phe",
    "cong truong",
    "cau",
    "ban hoc",
    "duong",
    "vach sang duong",
    "nha hang",
    "song",
    "cau go",
    "cua so",
    "phong hop",
    "san khau",
    "bep",
    "via he",
)

ACTION_PATTERNS = (
    ("mo cua", "cua"),
    ("mo khoa", "khoa"),
    ("buoc xuong", None),
    ("buoc ra ngoai", None),
    ("di vao", None),
    ("di qua", None),
    ("di bo", None),
    ("chay den", None),
    ("chay qua", None),
    ("nhan bong", "bong"),
    ("nem bong", "bong"),
    ("cam micro", "micro"),
    ("cam", None),
    ("chup anh", None),
    ("cat dien thoai", "dien thoai"),
    ("cat", None),
    ("lay chai nuoc", "chai nuoc"),
    ("lay quan ao", "quan ao"),
    ("lay mot cai banh", "banh"),
    ("lay hang", "hang"),
    ("lay", None),
    ("dua tui do", "tui do"),
    ("dua cho khach", "khach"),
    ("dua cho dua tre", "dua tre"),
    ("dua", None),
    ("dung truoc", None),
    ("dung canh", None),
    ("do truoc", None),
    ("xep hang", None),
    ("nam duoi", None),
    ("ngoi", None),
    ("dat dia thuc an", "dia thuc an"),
    ("dat vali", "vali"),
    ("dat sach", "sach"),
    ("dat", None),
    ("quet ma", "ma"),
    ("bat den", "den"),
    ("mo laptop", "laptop"),
    ("toi tram", "tram"),
    ("cua mo ra", "cua"),
    ("rua rau", "rau"),
    ("thai rau", "rau"),
    ("cho vao noi", "noi"),
    ("sap xep", None),
    ("dong cua", "cua"),
    ("deo day co", "day co"),
    ("cho xang", "xe"),
)

PERSON_TERMS = (
    "nguoi dan ong",
    "nguoi phu nu",
    "nguoi cong nhan",
    "nguoi ban hang",
    "nguoi giao hang",
    "nhan vien",
    "hanh khach",
    "em be",
    "dua tre",
    "cau thu",
    "dau bep",
    "khach",
    "nguoi",
)

VEHICLE_TERMS = ("xe buyt", "xe may", "xe tai", "xe", "tau hoa", "thuyen")


def parse_query_plan(
    query: str,
    intent: IntentClassification | None = None,
) -> QueryPlan:
    """Parse a user query into a validated ``QueryPlan``.

    ``intent`` can be passed in from Day 2 to avoid classifying twice. If it is
    omitted, this function calls ``classify_intent`` internally.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    resolved_intent = intent or classify_intent(query)
    task_type = resolved_intent.task_type
    normalized_query = normalize_text(query)

    if task_type == TaskType.QA:
        search_description, question = _split_qa_query(normalized_query)
        events: list[dict[str, object]] = []
    elif task_type == TaskType.TRAKE:
        search_description = _strip_search_prefix(normalized_query)
        question = None
        events = [
            {"event_id": index, "description": event}
            for index, event in enumerate(_split_trake_events(search_description), start=1)
        ]
    else:
        search_description = _strip_search_prefix(normalized_query)
        question = None
        events = []

    objects = _extract_keywords(search_description, OBJECT_KEYWORDS)
    actions = _extract_actions(search_description)
    scene = _extract_keywords(search_description, SCENE_KEYWORDS)
    entities = _extract_entities(objects, actions)

    raw_plan = {
        "task_type": task_type.value,
        "search_description": search_description,
        "question": question,
        "events": events,
        "entities": [entity.model_dump() for entity in entities],
        "objects": objects,
        "actions": actions,
        "scene": scene,
        "positive_constraints": _build_positive_constraints(objects, actions, scene),
        "negative_constraints": [],
        "metadata_keywords": _extract_metadata_keywords(normalized_query),
        "confidence": resolved_intent.confidence,
    }
    return validate_query_plan(raw_plan)


def _strip_search_prefix(text: str) -> str:
    for prefix in SEARCH_PREFIXES:
        if text.startswith(prefix + " "):
            return text[len(prefix) :].strip(" .")
    return text.strip(" .")


def _split_qa_query(text: str) -> tuple[str, str]:
    cleaned = text.strip(" .")
    question = cleaned if cleaned.endswith("?") else cleaned + "?"
    without_question_mark = cleaned.rstrip("?").strip()

    for prefix in QA_CONTEXT_PREFIXES:
        if without_question_mark.startswith(prefix + " "):
            body = without_question_mark[len(prefix) :].strip()
            if "," in body:
                context, ask = body.rsplit(",", 1)
                return context.strip(" ."), ask.strip(" .") + "?"

    if "," in without_question_mark:
        context, ask = without_question_mark.rsplit(",", 1)
        return _strip_search_prefix(context.strip()), ask.strip(" .") + "?"

    return _remove_question_words(without_question_mark), question


def _remove_question_words(text: str) -> str:
    replacements = (
        "bao nhieu",
        "mau gi",
        "la gi",
        "vat gi",
        "cai gi",
        "chu gi",
        "so gi",
        "loai gi",
        "o dau",
        "nhu the nao",
        "co khong",
        "hay khong",
        "bang tay nao",
        "ben trai hay ben phai",
        "phuong tien nao",
    )
    result = text
    for phrase in replacements:
        result = result.replace(phrase, "")
    return re.sub(r"\s+", " ", result).strip(" ,.")


def _split_trake_events(text: str) -> list[str]:
    parts = [
        part.strip(" .")
        for part in TEMPORAL_SPLIT_PATTERN.split(text)
        if part.strip(" .")
    ]
    return parts or [text.strip(" .")]


def _extract_keywords(text: str, keywords: Iterable[str]) -> list[str]:
    found = [keyword for keyword in keywords if _contains_phrase(text, keyword)]
    return _dedupe_preserve_order(found)


def _extract_actions(text: str) -> list[str]:
    actions = [action for action, _target in ACTION_PATTERNS if _contains_phrase(text, action)]
    return _dedupe_preserve_order(actions)


def _extract_entities(objects: list[str], actions: list[str]) -> list[Entity]:
    entities: list[Entity] = []
    for item in objects:
        entity_type = _infer_entity_type(item)
        entity_actions: list[EntityAction] = []
        if entity_type == "person":
            entity_actions = [
                EntityAction(verb=action, target=_infer_action_target(action, objects))
                for action in actions
            ]
        entities.append(Entity(name=item, type=entity_type, actions=entity_actions))

    if not entities and actions:
        entities.append(
            Entity(
                name="unknown",
                type="unknown",
                actions=[
                    EntityAction(verb=action, target=_infer_action_target(action, objects))
                    for action in actions
                ],
            )
        )
    return entities


def _infer_entity_type(name: str) -> str:
    if name in PERSON_TERMS:
        return "person"
    if name in VEHICLE_TERMS:
        return "vehicle"
    if name in SCENE_KEYWORDS:
        return "location"
    return "object"


def _infer_action_target(action: str, objects: list[str]) -> str | None:
    explicit_target = next(
        (target for known_action, target in ACTION_PATTERNS if known_action == action),
        None,
    )
    if explicit_target:
        return explicit_target
    return next((item for item in objects if item not in PERSON_TERMS), None)


def _build_positive_constraints(
    objects: list[str],
    actions: list[str],
    scene: list[str],
) -> list[str]:
    return _dedupe_preserve_order([*objects, *actions, *scene])


def _extract_metadata_keywords(text: str) -> list[str]:
    keywords = []
    for item in ("bien so", "bien hieu", "logo", "chu", "man hinh"):
        if _contains_phrase(text, item):
            keywords.append(item)
    return keywords


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped
