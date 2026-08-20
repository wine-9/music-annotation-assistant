# Evaluation & Calibration

生成时间：2026-07-31 16:38:29 +0800

## 结论

在本次真实公开音乐留出评估的 1575 个可判定标签字段中，满足逐类别 precision ≥ 0.85 的自动确认覆盖率为 **13.1%**（207 个字段）。
这些自动确认的整体 precision 为 **93.7%**；覆盖了真实正标签的 **24.5%**。
candidate coverage 为 **26.2%**，confirmed + candidate 总预填覆盖率为 **39.4%**，unknown coverage 为 **60.6%**。
按目标表单 10 个字段计算，公开数据能客观验证的 instrument_name 预填占全部目标字段的保守覆盖率为 **3.9%**。其余字段没有公开真值，不把 unknown 计作已覆盖。

> 这是工程规模、可复现的校准基准，不是生产精度承诺。阈值只改变决策层，模型和 Provider 均未修改。

## 数据来源与数量

- OpenMIC-2018：官方 test split 的 10 秒公开音乐片段；标签为人工部分标注，仅使用 mask 明确判定的正/负标签。数据集整体为 CC BY 4.0。
- MTG-Jamendo instrument subset：官方 split-0 test 元数据；下载的每首音乐只保留 12 秒评估片段，原始音频采用各曲目的 Creative Commons 许可。元数据为 CC BY-NC-SA 4.0，数据集限非商业研究用途。
- OpenMIC 不区分 electric/acoustic guitar，因此这两个类别只由 MTG-Jamendo 评估；OpenMIC 的 strings、brass、woodwind 由相应细类合并，只有明确标注时才计入。

| 数据集 / 分区 | 音频数 | 可判定标签字段 |
|---|---:|---:|
| mtg_jamendo_instrument / calibration | 208 | 400 |
| mtg_jamendo_instrument / evaluation | 482 | 932 |
| openmic_2018 / calibration | 0 | 0 |
| openmic_2018 / evaluation | 516 | 643 |

## 产品决策覆盖率

| 范围 | Confirmed | Candidate | Unknown | Confirmed coverage | Candidate coverage | 总预填覆盖 | Confirmed precision | Candidate precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 全部 | 207 | 413 | 955 | 13.1% | 26.2% | 39.4% | 93.7% | 63.9% |
| mtg_jamendo_instrument | 106 | 247 | 579 | 11.4% | 26.5% | 37.9% | 92.5% | 60.3% |
| openmic_2018 | 101 | 166 | 376 | 15.7% | 25.8% | 41.5% | 95.0% | 69.3% |

## 每个类别的产品决策数量

| 类别 | Confirmed | Candidate | Unknown | Confirmed precision | Candidate precision | 总预填覆盖 |
|---|---:|---:|---:|---:|---:|---:|
| drums | 0 | 94 | 96 | 0.0% | 76.6% | 49.5% |
| percussion | 0 | 5 | 145 | 0.0% | 80.0% | 3.3% |
| bass | 36 | 78 | 75 | 91.7% | 48.7% | 60.3% |
| electric_guitar | 0 | 49 | 51 | 0.0% | 71.4% | 49.0% |
| acoustic_guitar | 0 | 30 | 70 | 0.0% | 76.7% | 30.0% |
| piano | 66 | 40 | 77 | 97.0% | 47.5% | 57.9% |
| strings | 71 | 0 | 100 | 91.5% | 0.0% | 41.5% |
| brass | 0 | 0 | 154 | 0.0% | 0.0% | 0.0% |
| woodwind | 0 | 14 | 137 | 0.0% | 92.9% | 9.3% |
| synth | 34 | 103 | 50 | 94.1% | 58.3% | 73.3% |

## 留出评估结果与推荐阈值

下表“自动确认”是本次扩展评估产生的下一版建议，不会自动修改已审核的 `calibrated_public_v1`。当前 v1.0 仍只启用 bass、piano、strings、synth；例如 woodwind 的新建议需人工审核并另发配置版本后才会生效。

| 类别 | N（正/负） | 推荐阈值 | Precision | Recall | F1 | PR-AUC | 当前阈值覆盖 | 推荐覆盖 | 自动确认 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| drums | 190（100/90） | 0.9207 | 0.0% | 0.0% | 0.0% | 0.803 | 0.0% | 0.0% | 禁用 |
| percussion | 150（56/94） | 1.0000 | 0.0% | 0.0% | 0.0% | 0.600 | 0.0% | 0.0% | 禁用 |
| bass | 189（93/96） | 0.5538 | 91.7% | 35.5% | 51.2% | 0.788 | 19.0% | 19.0% | 启用 |
| electric_guitar | 100（50/50） | 0.3684 | 67.6% | 46.0% | 54.8% | 0.705 | 34.0% | 34.0% | 禁用 |
| acoustic_guitar | 100（50/50） | 0.1329 | 68.3% | 82.0% | 74.5% | 0.788 | 60.0% | 60.0% | 禁用 |
| piano | 183（93/90） | 0.7095 | 97.0% | 68.8% | 80.5% | 0.931 | 36.1% | 36.1% | 启用 |
| strings | 171（96/75） | 0.1572 | 91.5% | 67.7% | 77.8% | 0.914 | 41.5% | 41.5% | 启用 |
| brass | 154（61/93） | 0.0280 | 62.9% | 91.8% | 74.7% | 0.795 | 57.8% | 57.8% | 禁用 |
| woodwind | 151（97/54） | 0.1332 | 91.8% | 57.7% | 70.9% | 0.897 | 40.4% | 40.4% | 启用 |
| synth | 187（97/90） | 0.7098 | 94.1% | 33.0% | 48.9% | 0.834 | 18.2% | 18.2% | 启用 |

## 校准方法

1. 使用 MTG-Jamendo calibration 子集搜索每类阈值。
2. 枚举真实模型分数阈值，只保留 precision ≥ 0.85 的候选。
3. 在合格候选中选择预测正例数最多、即标签字段覆盖率最高的阈值。
4. 验证正样本少于 20 时强制关闭 auto-confirm。
5. 在未参与阈值搜索的 evaluation 子集上复核；留出 precision 未达到 0.85 的类别保持禁用。

Coverage 定义为达到 confirmed threshold 的标签字段数除以全部可判定标签字段数。PR-AUC 使用 non-interpolated average precision。

## 风险与限制

- MTG-Jamendo 是上传者提供的弱标签；标签缺失被当作负例时可能包含漏标。
- 当前模型头本身在 MTG-Jamendo instrument 数据上训练，MTG 结果属于同分布评估；OpenMIC 用于补充跨数据集检查。
- 每首 MTG 音乐只抽取固定 12 秒，可能没有覆盖元数据所标注乐器真正出现的段落，因此结果偏保守。
- 样本规模适合工程校准，不足以替代完整数据集上的正式论文级评测。
- 推荐阈值写入 `outputs/evaluation/recommended_thresholds.yaml`，不会自动覆盖生产配置。

## 可复现命令

```bash
make setup-evaluation
make download-evaluation
make evaluate
```

机器可读结果：`outputs/evaluation/report.json`；逐样本原始分数：`outputs/evaluation/predictions.csv`。

## 来源

- OpenMIC-2018：https://zenodo.org/records/1432913
- MTG-Jamendo：https://mtg.github.io/mtg-jamendo-dataset/
- MTG-Jamendo 官方仓库：https://github.com/MTG/mtg-jamendo-dataset
