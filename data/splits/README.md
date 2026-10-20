# data/splits

`analysis/make_splits.py` 生成的 train / val / test 切分（已入库，全组用同一份）。

## 文件

| 文件 | 内容 |
|---|---|
| `train.csv` | 606 行（202 个 `id`），三类各 202 行 |
| `val.csv` | 132 行（44 个 `id`），三类各 44 行 |
| `test.csv` | 132 行（44 个 `id`），三类各 44 行 |
| `id_to_split.csv` | 清单：每个 `id` 属于哪个 split、行数、包含的标签 |
| `split_summary.json` | 规模、每类行数、每 split 的 id 清单 sha256 |

## 关键设定

- **按 `id` 分组切分**：870 行来自 290 个受访者，每个 `id` 有 3 行（Soccer / Baseball / Basketball 各一行）。
  同一个 `id` 的所有行必须落在同一个 split，否则同一个人会同时出现在 train 和 test，指标虚高。
- **比例 70 / 15 / 15**（按分组计）：train 202 组、val 44 组、test 44 组。
- 因为每个 `id` 恰好覆盖 3 个类，**每个 split 内三类严格均衡**（202/202/202、44/44/44、44/44/44），无需再分层。
- **切分文件逐字节保留原始文本**：读入时用 `dtype=str`，所以 `class_dataset.csv` 里的 `5` 不会变成 `5.0`，
  脚本会回读三个文件并与源数据做强校验。

## 可重现性

切分顺序由 `sha256("<seed>:<id>")` 决定，不用随机数发生器，因此与 numpy 版本、平台、CSV 行序都无关：

```bash
conda activate csc311
python analysis/make_splits.py                      # 默认 seed=311, 0.15 / 0.15
python analysis/make_splits.py --seed 7 --test-size 0.1 --val-size 0.1
python analysis/make_splits.py --from-manifest data/splits/id_to_split.csv   # 直接复用清单
```

同一条命令重复运行会得到**完全相同**的文件（已验证：5 个输出文件的 sha256 全部一致）。

## 使用约定

- **任何特征工程 / 调参 / 统计只能看 `train.csv`**（当前 `analysis/out/train/` 里的分析就是 train 上的）；
- `val.csv` 用于模型选择与超参比较，`test.csv` 只在最终评估时用一次；
- `data/class_dataset_tfidf.csv` 是在**全量 870 行**上拟合 TF-IDF 得到的，属于词频泄漏，
  后续要用它建模必须在 train 上重新 `fit`，或直接改用 train 重新生成。
