# data_visualization

所有图表由 `plot_histograms.py` 生成，输出到 `figures/`。图表内容：**train 集里每一项特征在 3 个类
（Soccer / Baseball / Basketball）内部的分布**，以分组直方图表示，每根柱子上标出该类人数（多选题还标类内占比）。

> **数据口径：只用 train 集。** 脚本默认读 `data/splits/train.csv`（606 行 / 202 个 `id`）。
> 因为切分是按 `id` 分组的，val 和 test 不能参与任何描述性分析或画图。

> **图上不放表格。** 逐类的均值、方差、标准差以及各类取值分布等统计结果全部放在 markdown 里：
> `analysis/out/train/report_single_multi.md`（train 集，含数据切分说明）。
> 图上只保留一行紧凑的 `mean:` / `variance:`（顺序为 Soccer / Baseball / Basketball）方便读图。

**图内所有文字为英文。** 每张图的标题格式为 `Q编号 | 完整题干`，第二行给出题型，例如：

```
Q3 | Players must be comfortable with a high risk of injury.
Single choice, Likert 1-5 (ordinal score 1-5)   |   Soccer / Baseball / Basketball
mean: 3.83 / 2.92 / 3.85      variance: 0.85 / 1.02 / 0.82
```

## 运行

```bash
conda activate csc311
python data_visualization/plot_histograms.py                       # 默认 train 集
# 可选：换数据或输出位置
python data_visualization/plot_histograms.py --data data/splits/val.csv --out-dir data_visualization/figures
```

依赖：`pandas`、`matplotlib`、`Pillow`（均已在 `csc311` 环境中）。

## 图文件

| 文件 | 内容 |
|---|---|
| `figures/00_overview.png` | 总览：上排 3 道单选题的 5 级直方图，下排 3 道多选题的「勾选个数」直方图 |
| `figures/01_single_Q2.png` | Q2 策略与规划：5 级序数分布（三类分组） |
| `figures/02_single_Q3.png` | Q3 受伤风险：5 级序数分布（三类分组） |
| `figures/03_single_Q4.png` | Q4 个人能力胜团队：5 级序数分布（三类分组） |
| `figures/04_multi_Q5.png` | Q5 场地：7 个选项被三类各自勾选的人数与类内占比 |
| `figures/05_multi_Q6.png` | Q6 季节：4 个选项 |
| `figures/06_multi_Q7.png` | Q7 音乐类型：10 个选项 |
| `figures/07_count_Q5.png` | Q5 场地勾选个数（0–7）的分布 |
| `figures/08_count_Q6.png` | Q6 季节勾选个数（0–4）的分布 |
| `figures/09_count_Q7.png` | Q7 音乐类型勾选个数（0–10）的分布 |

## 编码口径

- 单选题 Likert → 序数 1–5：`Strongly disagree=1, Disagree=2, Neutral=3, Agree=4, Strongly agree=5`
- 多选题 → 每个选项一个 0/1 变量（勾选=1），柱高=该类别中勾选此项的人数
- 多选题的 0/1 变量方差 = `n/(n-1)·p·(1-p)`，由均值决定，图上脚注已注明
- 空回答在统计时剔除，各类实际 n 标在图例中
- 读入 CSV 使用 `keep_default_na=False`：否则真实回答 `None` / `N/A` 会被 pandas 当成缺失值

## 配套的 markdown 统计

`analysis/categorical_stats.py` 生成 `analysis/out/train/report_single_multi.md`，包含：

- 数据切分说明（train / val / test 各自行数、分组数、占比、切分方式）
- 每道单选题的逐类 `n / 均值 / 方差 / 标准差` 与 5 级取值分布
- 每个多选题选项 × 每一类的勾选率与方差、χ²、Cramér's V
- 「勾选个数」特征的逐类统计
- 按效应量（Cramér's V / η²）排序的特征重要性表
