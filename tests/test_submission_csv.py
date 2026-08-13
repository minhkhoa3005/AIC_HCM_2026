from pipeline.prediction import Prediction
from submission.config import SubmissionConfig
from submission.csv_exporter import export_csv
from llm.task_types import TaskType


def test_export_csv_quotes_answers_and_applies_frame_config(tmp_path):
    path = tmp_path / "submission.csv"
    predictions = [
        Prediction(
            query_id="Q001",
            task_type=TaskType.QA,
            rank=1,
            video_id="L01_V001",
            frame_ids=[0, 9],
            answer="đỏ, trắng",
        )
    ]

    export_csv(
        predictions,
        path,
        SubmissionConfig(frame_base=1, include_mp4_extension=True),
    )

    assert path.read_text(encoding="utf-8").splitlines() == [
        "query_id,rank,video_id,frame_ids,answer",
        'Q001,1,L01_V001.mp4,1 10,"đỏ, trắng"',
    ]
