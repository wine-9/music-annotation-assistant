# 架构

音频先经 `AudioLoader` 生成两条数据路径：mono 16 kHz 供模型使用，原采样率 stereo
供 DSP 使用。`EssentiaInstrumentProvider` 是唯一接触 Essentia API 的模块，业务层
只依赖 `InstrumentProvider` 协议。

模型逐时间帧输出官方 40 类 Sigmoid 分数。Decision Engine 保留原始分数，并按
taxonomy 将多个原始标签聚合到规范化标签，计算完整统计量、存在分数和时间区间。
所有阈值从只读 YAML 档案读取。运行时只保存档案名称，可在 `default` 与
`calibrated_public_v1` 间回滚，不覆盖原始配置。互斥细分类得分接近时只保留候选，
不强行细分。

Spatial Analyzer 默认只分析整体混音。可选 Demucs 仅提供 drums、bass、vocals、
other 四轨；只有语义匹配且能量可靠的 drums/bass/vocals Stem 可以参与保守融合，
`other` 不用于直接认定吉他、钢琴或弦乐。目标表单映射禁止把整体混音声场复制给
单乐器。

任务、状态和路径保存在本地 SQLite，结构化结果保存在独立 JSON；人工修改和
Human Review 以追加事件形式记录，保留原始预测、最终值、档案版本及真实操作时间。
