# DAS 轨道结构异常检测：可控基准与负结果

基于分布式声学传感（DAS）的轨道结构异常无监督检测研究。仓库里放的是**可复现的评测基准**：
物理一致的数据注入、严格标定的信噪比、指标隔离、种子纪律，以及在这些约束下对
多种检测范式的比较。

> **当前状态一句话**：在自建仿真上，一个零参数的 80–160 Hz 带通包络检测器
> 在绝大多数条件下不劣于任何学习模型；换掉缺陷家族后，学习模型的优势完全消失。
> 项目的主线因此从"造一个更好的网络"改为"搞清什么条件下什么方法成立"。

详细报告见 [`reports/snr_benchmark.md`](reports/snr_benchmark.md)（9 节）。
所有被引用的数字都能在 [`results/metrics_snapshot.csv`](results/metrics_snapshot.csv)
里查到原始记录。

## 这个基准解决了什么

| 问题 | 做法 |
|---|---|
| 信噪比不可控 | 加性物理注入（冲击 / 阻尼共振 / 扫频 / 周期冲击 / 调幅），预处理后实测局部 SNR，定点迭代标定（精度 0.049 dB） |
| 指标互相混淆 | 窗内**通道级**与**过车级**分开报，避免车轴能量污染 |
| 单次运行当结论 | 多种子 + 配对符号检验；**AUC 差值 < 0.04 一律不算结论** |
| 阈值不可复现 | 独立标定集 + 共形预测 + 过车级阈值；给出样本量规律 **n_cal ≈ 5/α** |
| 结论只对自己设计的缺陷成立 | 四种家族外形态 + 三种强经典对手（匹配滤波 / 盲滤波器组 / 自适应选带） |

## 主要发现

1. **固定带通很难被打败**。通道级 0.749、过车级 0.882，高于 1D AE（0.710）、
   掩蔽 AE（0.40–0.57）、STFT AE（0.744）。
2. **基于频带的学习表示只在"同族频率漂移"这一格赢**。抖动 0.7 时 `stft_ae`
   过车级 +0.088（5/5 同号），但换到扫频 / 周期冲击 / 调幅后一次都没赢过
   （`impulse_train` 上 −0.188，3/3 同号）。
3. **自适应选带全部输给不动的频带**。谱峭度 0.575、最响带 0.543 vs 带通 0.882。
   机制是时间尺度：缺陷带在一条过车之内恒定，逐窗选带把这份一致性打碎。
4. **"换成时会看时变结构"的补救也被证伪**。包络谱与时频轨迹检测器在它们最该赢的
   `impulse_train` / `chirp` 上都只有 0.45–0.55，且系统性低于随机。
5. **阈值稳定性是工程瓶颈，不是精度问题**。n=21、α=0.05 时共形分位数就是样本最大值，
   实测 31–38% 的情况下永不报警。

## 目录

```
configs/            实验配置（SNR 扫描、抖动、族外形态、标定规模）
src/data_io/        仿真器：物理注入、多站、过车切片
src/models/         bandpass / filterbank / pca / ae_1d / masked_ae / stft_ae /
                    kurtogram / bandselect / envspec / tftraj
src/evaluation/     共形阈值、过车级统计、区间定位
scripts/            train / evaluate / run_snr_sweep / calibration_size_curve /
                    analyze_morphology_seeds / attribute_frequency /
                    export_metrics_snapshot
reports/            主报告
results/            指标快照（CSV）
```

`data/` 与 `outputs/` 不进版本库：前者是现场数据入口，后者是可由 config + seed
重建的产物。

## 复现

```bash
python scripts/train.py --config configs/exp_default.yaml --model bandpass
python scripts/evaluate.py --config configs/exp_default.yaml --model bandpass
python scripts/run_snr_sweep.py --config configs/exp_snr_sweep.yaml \
        --snr -5 0 5 --repeats 3 --models bandpass stft_ae
python scripts/export_metrics_snapshot.py
```

仿真与指标有单元级自检：`python scripts/verify_simulator.py`。

## 未完成 / 下一步

- **真实背景注入**：把物理注入叠到真实采集的 DAS 背景上（不需要缺陷标注），
  用来检验"噪声太干净"这个批评。当前结论全部建立在自建仿真背景上。
- **更细的带集**：逐过车选带只测了 `filterbank` 的 5 个 30 Hz 带宽（60–240 Hz），
  谱峭度的 13 带没有覆盖。但 oracle 上界已经压到 0.58，预计不会翻盘。
- **扫出通带的 chirp**：现有 `chirp` 的 60–180 Hz 与带通 80–160 Hz 高度重叠，
  对带通并不构成失配。
- **现场闭环**：待真实线路数据到位后开展（与工务台账、轨检记录比对）。

## 已排除的路线（四条，均为独立实验）

| 尝试 | 结果 |
|---|---|
| 逐窗自适应选带（谱峭度 / 最响带） | 输给跨带平均 0.17–0.32 |
| 换学习表示（STFT 对数幅度谱） | 只在同族抖动下赢，换形态即失效 |
| 看时变结构的检测器（包络谱 / 时频轨迹） | 0.44–0.55，系统性低于随机 |
| 逐过车选带（含用标签作弊的上界） | 最好的无标签变体 0.585，作弊 0.581，取平均 0.695 |

四条指向同一结论：**"怎么聚合"比"用什么表示"重要，而聚合的最佳形式是平均，不是选择。**

## 说明

所有实验均为仿真，结论的适用范围以报告中标注的条件为准。仓库不包含任何现场数据。
