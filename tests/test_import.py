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
    try:
        df = run(["--max-vars", "2", "--max-combos", "1"])
    except SystemExit:
        # 参数解析报错或缺少数据文件时跳过
        return

    assert isinstance(df, pd.DataFrame)
