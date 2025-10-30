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

### 任务示例
- **变量筛选**（数据驱动）  
  ```bash
  python -m robot_analysis \
    --task select \
    --groups macro,industry,innovation \
    --max-controls 8 \
    --selection-method lasso \
    --verbose \
    --output results_controls.csv
  ```
- **基准估计**（双重固定效应，`x` 滞后 1 期）  
  ```bash
  python -m robot_analysis \
    --task estimate \
    --lag-x 1 \
    --base-vars "log_gdp,manufacturing_share" \
    --output results_baseline.csv
  ```
- **机制与门槛检验**  
  ```bash
  python -m robot_analysis \
    --task mechanism \
    --mediator "R&D研究人员 （每百万人）" \
    --moderator "制造业，增加值（占GDP的百分比）"

  python -m robot_analysis \
    --task threshold \
    --threshold-var "制造业出口（占商品出口的百分比）" \
    --threshold-bootstrap 200
  ```
- **稳健性分析**（口径/滞后/插补对比）  
  ```bash
  python -m robot_analysis \
    --task robustness \
    --lag-x 1 \
    --output results_robustness.csv
  ```

常用参数（不同任务共享）：
- `--task {select,estimate,mechanism,threshold,robustness}`：指定要执行的分析步骤。
- `--groups`：候选变量分组；`--extra-vars`、`--base-vars` 可补充/固定特定变量。
- `--missing-threshold` / `--corr-alpha` / `--vif-threshold`：控制筛选阶段的缺失率、相关性与共线性阈值。
- `--selection-method {lasso,stepwise}`、`--max-controls`：控制变量筛选策略与最大数量。
- `--lag-x`：`x` 的滞后期数；机制/门槛任务会复用该设定。
- `--threshold-bootstrap`：门槛检验的 Bootstrap 次数。
- `--output`：将结果导出为 CSV/JSON。

更多选项详见 `python -m robot_analysis --help`。

## 开发与测试
```bash
pip install -e .[dev]
pytest
```

## 推广建议
- 在 Jupyter Notebook 中调用 `from robot_analysis import run` 以便进行可视化扩展。
- 将结果 CSV 作为分析记录纳入项目文档或论文附录。
- 可在 `src/robot_analysis/cli.py` 中自定义变量分组或扩展建模策略。*** End Patch
