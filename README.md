# Robot Analysis

标准化的 Python 项目，用于探索机器人渗透率 (`x`) 与全球价值链地位 (`y`) 间的关系，并帮助研究者快速筛选高拟合度的控制变量组合。

## 项目结构
```
.
├── analysis_plan.md         # 研究思路与变量筛选流程说明
├── data/                    # 原始数据及后续衍生数据存放目录
│   └── data.xlsx
├── pyproject.toml           # 包与依赖配置，支持 `pip install .`
├── requirements.txt         # 运行所需基础依赖
├── src/robot_analysis/      # 包代码（包含 CLI 入口）
└── tests/                   # 基本可用性测试
```

## 环境准备
1. （可选）创建虚拟环境：
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   # 或者：pip install -e .
   ```

## 命令行用法
数据默认读取 `data/data.xlsx`，如需指定其它文件，通过 `--data` 参数传入。

```bash
python -m robot_analysis \
  --method stepwise \
  --groups macro,industry,innovation \
  --max-vars 6 \
  --standardize \
  --data data/data.xlsx \
  --output results_stepwise.csv
```

常用参数说明：
- `--method {stepwise,subsets}`：选择变量搜索策略。
- `--groups`：指定候选变量分组，多个分组用逗号分隔。
- `--base-vars`：强制纳入模型的控制变量（原始列名）。
- `--max-vars`、`--min-improvement`：限制模型复杂度与逐步回归增益。
- `--max-missing`：剔除缺失率超过阈值的变量。
- `--standardize`：对控制变量做标准化。
- `--output`：将结果导出为 CSV。

更多参数可通过 `python -m robot_analysis --help` 查看。

## 开发与测试
```bash
pip install -e .[dev]
pytest
```

## 推广建议
- 在 Jupyter Notebook 中调用 `from robot_analysis import run` 以便进行可视化扩展。
- 将结果 CSV 作为分析记录纳入项目文档或论文附录。
- 可在 `src/robot_analysis/cli.py` 中自定义变量分组或扩展建模策略。*** End Patch
