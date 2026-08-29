import csv
from loralink_reviewer_response.metrics_logger import append_rows, RUN_COLUMNS

def test_header_written_once_and_rows_appended(tmp_path):
    p = tmp_path / "m.csv"
    append_rows(p, [{"run_tag": "a", "seed": 0, "loss": 1.5}], RUN_COLUMNS)
    append_rows(p, [{"run_tag": "a", "seed": 1, "loss": 1.2}], RUN_COLUMNS)
    with open(p, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == RUN_COLUMNS            # header once
    assert rows[1][RUN_COLUMNS.index("loss")] == "1.5"
    assert rows[2][RUN_COLUMNS.index("seed")] == "1"
    assert len(rows) == 3

def test_missing_keys_become_blank_and_extra_keys_ignored(tmp_path):
    p = tmp_path / "m.csv"
    append_rows(p, [{"run_tag": "x", "nonsense": 9}], RUN_COLUMNS)
    with open(p, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[1][RUN_COLUMNS.index("run_tag")] == "x"
    assert rows[1][RUN_COLUMNS.index("loss")] == ""
