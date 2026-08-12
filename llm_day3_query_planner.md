# Day 3 - Query Planner

## Muc tieu

Day 3 noi Day 2 intent classifier voi phan retrieval CLIP.

Luong chinh:

```text
user query
  -> classify_intent(query)
  -> parse_query_plan(query, intent)
  -> build_clip_queries(QueryPlan)
  -> search_clip_text(list[str])
  -> list[Candidate]
```

## Module moi

### `llm/parser.py`

Tao `QueryPlan` tu cau query goc.

Ham chinh:

- `parse_query_plan(query, intent=None)`: ham chinh cua Day 3. Neu chua co intent thi goi `classify_intent`. Sau do tach `search_description`, `question`, `events`, `objects`, `actions`, `scene`, `entities`.
- `_split_qa_query(text)`: tach query QA thanh context tim kiem va cau hoi.
- `_split_trake_events(text)`: tach TRAKE thanh cac event theo thu tu.
- `_extract_keywords(text, keywords)`: lay object/scene bang keyword rule local.
- `_extract_actions(text)`: lay action bang rule local.
- `_extract_entities(objects, actions)`: tao entity nhe de giu quan he ai lam gi voi cai gi.

### `llm/query_builder.py`

Bien `QueryPlan` thanh input dung cho retrieval.

Ham chinh:

- `build_clip_queries(plan)`: tra ve `list[str]` de dua vao `search_clip_text`.
- `build_rerank_text(plan)`: tao text tom tat gon cho rerank/QA prompt ve sau.

## Vi sao can buoc nay?

`validate_query_plan` chi validate dict/raw JSON thanh object Pydantic. No khong tu hieu cau user.

Day 3 them parser de bien:

```text
Tim canh nguoi dan ong mac ao do mo cua xe mau trang.
```

thanh:

```python
QueryPlan(
    task_type=TaskType.TEXTUAL_KIS,
    search_description="nguoi dan ong mac ao do mo cua xe mau trang",
    objects=["nguoi dan ong", "xe", "cua xe"],
    actions=["mo cua"],
)
```

Sau do `build_clip_queries(plan)` bien `QueryPlan` thanh `list[str]`:

```python
[
    "nguoi dan ong mac ao do mo cua xe mau trang",
    "nguoi dan ong",
    "xe",
    "cua xe",
    "mo cua",
]
```

Day la ly do `search_clip_text` nhan `list[str]`: mot query user co the can nhieu cau search phu.

## Trang thai V1

Parser hien tai la local heuristic, chua goi API. Ve sau co the thay bang LLM parser/RAG parser, nhung output van nen giu `QueryPlan` de retrieval va rerank khong bi doi contract.
