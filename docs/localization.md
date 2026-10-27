# 定位与告警粒度说明

本文档说明 `scripts/evaluate.py` 中新增的整趟（pass）级检测、通道定位和异常区间指标。

## 为什么需要两套粒度

逐窗口指标把每个 `Tw=128` 的切片当成独立样本。实际 DAS 数据中，移动载荷背景在窗口之间变化很大，单窗口的 `argmax` 很容易落到背景通道。轨道缺陷则会在一趟过车的多个窗口中持续出现。

因此评估脚本同时输出：

| 粒度 | 输入 | 用途 |
|---|---|---|
| window | 每个窗口的归一化分数 | 观察单窗口检测能力，保留原有指标 |
| pass | 同一 `source` 的所有窗口 | 实际告警决策 |
| interval | 整趟融合后的通道曲线 | 输出需要检修的通道区间 |

## 整趟检测

```yaml
evaluation:
  source_score_aggregation: mean
  pass_threshold_margin: 0.01
```

训练时除窗口阈值外，还会计算每个正常 `source` 的平均分数，并保存：

```text
pass_threshold = max(normal_pass_scores) * (1 + pass_threshold_margin)
```

该阈值和 `pass_score_aggregation` 写入 checkpoint 元数据，评估时直接使用，避免拿测试集标签调阈值。

## 整趟通道定位

```yaml
evaluation:
  localization_aggregation: vote
```

`vote` 的含义是：每个窗口对自己的最强通道投一票，再统计每个通道的得票率。它只关心“哪个通道最强”，不受窗口之间绝对残差幅值变化的影响。定位输入使用 checkpoint 输出的原始（未逐通道归一化）通道残差。

如果设置为 `mean`、`median` 或 `max`，则对这些原始通道分数做对应聚合。

## 异常区间

```yaml
evaluation:
  localization_interval:
    threshold_mad: 1.75
    merge_gap: 2
    min_width: 1
    smoothing_sigma: 1.0
```

处理流程：

1. 对整趟融合后的通道曲线做高斯平滑；
2. 计算鲁棒基线 `median` 和 `MAD`；
3. 阈值 `threshold = median + threshold_mad * MAD`；
4. 将连续超阈值通道合并为区间；
5. `merge_gap` 允许区间内部有少量弱通道；
6. 取峰值最高的区间作为主异常区间。

输出指标：

- `hit_rate`：预测区间与真实区间是否有交集；
- `mean_iou`：区间交并比；
- `mean_coverage`：真实通道被预测区间覆盖的比例；
- `mean_start_error_channels` / `mean_end_error_channels`：区间起止边界误差；
- `mean_predicted_width_channels`：预测区间平均宽度；
- `exact_interval_rate`：起止完全一致的比例。

## 输出文件

运行：

```powershell
python scripts/evaluate.py --model pca
```

生成：

```text
outputs/predictions/metrics_pca.json       # 全部指标和逐 source 细节
outputs/predictions/intervals_pca.csv       # 每趟的预测/真实区间
outputs/predictions/source_scores_pca.csv   # 每趟检测分数
outputs/predictions/profile_pass_pca.png    # 整趟通道投票曲线与区间
```

`metrics_pca.json` 中的主要字段：

```json
{
  "threshold": 10.57,
  "pass_threshold": 12.09,
  "metrics": {
    "roc_auc": 0.83,
    "f1": 0.82,
    "detection_by_source": {"roc_auc": 1.0, "f1": 1.0},
    "localization_by_source": {"mean_error_channels": 0.0},
    "localization_intervals": {"hit_rate": 1.0, "mean_iou": 0.68}
  }
}
```

## 参数选择注意事项

- `threshold_mad` 越小，区间越宽；越大，区间越窄。应在有标注的验证集上选择，不要只看测试集。
- `vote` 适合“缺陷在整趟中持续存在”的场景；如果缺陷只出现在很短时间，应改用 `max` 或先做时间持久性筛选。
- 真实线路的通道间距、异常宽度和背景噪声与合成数据不同，部署前必须重新校准 `threshold_mad`、`merge_gap` 和 `pass_threshold_margin`。