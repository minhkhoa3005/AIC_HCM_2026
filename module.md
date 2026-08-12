# LLM Module Refactor Notes

## Muc Tieu

Module `llm` nen duoc tach theo trach nhiem ro rang de de bao tri, test va mo rong:

- Classifier chi quyet dinh `task_type`.
- Parser chi extract field cua `QueryPlan`.
- Planner chi dieu phoi luong chay.
- Provider an chi tiet Gemini/API ben duoi.
- Schema, validator va query builder tach khoi logic LLM.

Thiet ke mong muon:

```text
LLM classifier: duoc quyen quyet dinh task_type
LLM parser: khong quyet dinh task_type nua, chi extract fields
```

## Cau Truc De Xuat

```text
llm/
  __init__.py

  config.py

  core/
    __init__.py
    schemas.py
    task_types.py
    validator.py

  classifier/
    __init__.py
    local_classifier.py
    llm_classifier.py

  parser/
    __init__.py
    local_parser.py
    llm_parser.py

  planner/
    __init__.py
    planner.py
    complexity.py

  providers/
    __init__.py
    base.py
    gemini.py

  prompts/
    __init__.py
    classifier_prompt.py
    parser_prompt.py

  query/
    __init__.py
    builder.py
```

## Trach Nhiem Tung Module

### `core/`

Chua cac thanh phan nen tang, khong phu thuoc classifier/parser:

```text
schemas.py
task_types.py
validator.py
```

`core` nen la noi dinh nghia contract chinh cua pipeline:

- `TaskType`
- `IntentClassification`
- `QueryPlan`
- `validate_query_plan`

### `classifier/`

Chi lam nhiem vu phan loai truy van thuoc task nao.

```text
classifier/
  local_classifier.py
  llm_classifier.py
```

Output cua classifier chi nen la:

```python
IntentClassification(task_type, confidence, reason)
```

`local_classifier`:

- Chay offline.
- Nhanh, re, deterministic.
- Dung lam buoc dau tien va fallback.

`llm_classifier`:

- Chi duoc goi khi local classifier confidence thap va env cho phep.
- Duoc quyen override `task_type`.
- Khong extract `actions`, `entities`, `objects`, `events`.

### `parser/`

Chi boc tach field cho `QueryPlan`.

```text
parser/
  local_parser.py
  llm_parser.py
```

Parser nhan `IntentClassification` da duoc quyet dinh tu classifier.

`llm_parser` khong duoc quyet dinh lai `task_type`. Neu LLM tra ve `task_type` khac, code nen ep ve `final_intent.task_type` truoc khi validate.

Parser chi nen extract cac truong nhu:

- `actions`
- `entities`
- `objects`
- `events`
- `scene`
- `temporal`
- cac field lien quan den `QueryPlan`

### `planner/`

Chua orchestration cua pipeline.

```text
planner/
  planner.py
  complexity.py
```

`planner.py` nen chua entrypoint chinh:

```python
plan_query(query)
```

`complexity.py` chua gate quyet dinh co can dung LLM parser hay khong:

```python
is_complex_query(query, intent)
should_use_llm_parser(query, intent, config)
```

### `providers/`

An chi tiet model/API ben duoi.

```text
providers/
  base.py
  gemini.py
```

Code ben ngoai khong nen import truc tiep `gemini_parser` hay `gemini_classifier`.

Ten public nen la generic:

```text
llm_parser
llm_classifier
llm_client
```

Gemini chi nen la implementation trong provider:

```text
providers/gemini.py
```

Dieu nay giup sau nay doi model Gemini hoac them provider khac ma khong anh huong den classifier/parser/planner.

### `prompts/`

Tach prompt ra khoi logic code.

```text
prompts/
  classifier_prompt.py
  parser_prompt.py
```

Prompt classifier:

- Yeu cau LLM tra ve `task_type`, `confidence`, `reason`.
- Khong yeu cau extract `QueryPlan`.

Prompt parser:

- Noi ro `task_type` da duoc classify truoc.
- LLM parser phai dung dung `task_type` duoc cung cap.
- Chi extract fields cua `QueryPlan`.

### `query/`

Chua cac ham build query phuc vu retrieval/search.

```text
query/
  builder.py
```

Vi du:

```python
build_clip_queries(plan)
build_rerank_text(plan)
```

## Luong Chay A-Z

```text
A. User query
   ->
B. planner.plan_query(query)
   ->
C. classifier.local_classifier.classify_intent(query)
   ->
D. Neu local confidence thap:
      USE_LLM_CLASSIFIER=true
      confidence < INTENT_CONFIDENCE_THRESHOLD
      -> classifier.llm_classifier.classify_intent_with_llm(query)
   ->
E. Co final_intent chinh thuc
   ->
F. planner.complexity.should_use_llm_parser(query, final_intent, config)
   ->
G1. Neu khong dung LLM parser:
      parser.local_parser.parse_query_plan(query, final_intent)

G2. Neu dung LLM parser:
      parser.llm_parser.parse_query_plan_with_llm(query, final_intent)
      LLM parser chi extract fields, khong override task_type
   ->
H. core.validator.validate_query_plan(plan)
   ->
I. query.builder.build_clip_queries(plan)
   ->
J. search_clip_text(...)
```

## Quyet Dinh Dung LLM

### LLM Classifier

LLM classifier duoc dung khi:

```text
USE_LLM_CLASSIFIER=true
va local_intent.confidence < INTENT_CONFIDENCE_THRESHOLD
```

Muc dich:

- Sua lai `task_type` khi local classifier khong chac.
- Giam rui ro parser phai vua classify vua extract.

### LLM Parser

LLM parser duoc dung khi:

```text
USE_LLM_PARSER=true
```

Va tuy theo mode:

```text
LLM_PARSER_MODE=always
  -> luon dung LLM parser

LLM_PARSER_MODE=auto
  -> chi dung khi query phuc tap hoac confidence thap

LLM_PARSER_MODE=never
  -> khong dung LLM parser
```

`should_use_llm_parser` la noi quyet dinh co goi LLM parser hay khong.

## Env De Xuat

```env
LLM_PROVIDER=gemini
LLM_API_KEY=
LLM_MODEL=gemini-2.5-flash

USE_LLM_CLASSIFIER=false
LLM_CLASSIFIER_TEMPERATURE=0
INTENT_CONFIDENCE_THRESHOLD=0.8

USE_LLM_PARSER=false
LLM_PARSER_MODE=auto
LLM_PARSER_TEMPERATURE=0
```

Khong can `OPENAI_API_KEY` neu chi dung Gemini.

## Nguyen Tac Can Giu

- Khong de `gemini_parser` / `gemini_classifier` xuat hien trong public flow.
- Khong de parser quyet dinh lai `task_type`.
- LLM output luon phai qua validator.
- Local classifier/parser phai tiep tuc ton tai de fallback va test offline.
- Unit test khong duoc goi API that.
- Default phai an toan chi phi:

```text
USE_LLM_CLASSIFIER=false
USE_LLM_PARSER=false
```
