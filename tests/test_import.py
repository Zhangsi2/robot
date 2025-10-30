from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def test_run_returns_dataframe():
    import pandas as pd

    from robot_analysis import run

    # 仅测试函数可调用且返回 DataFrame，不要求具体结果
    df = run(["--task", "select", "--groups", "macro", "--max-controls", "2"])

    assert isinstance(df, pd.DataFrame)
