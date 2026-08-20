# macOS / Apple Silicon 设置

推荐 Python 3.12。项目环境与系统 Python 隔离。FFmpeg 由 `imageio-ffmpeg` 在项目
依赖内提供，因此无需修改系统安装。

```bash
make setup PYTHON_BOOTSTRAP=/path/to/python3.12
make setup-model
make download-models
make verify-models
make test
make run
```

如 Essentia 安装失败，检查 Python/arm64/macOS wheel 是否匹配，不要改写官方预处理
或使用模拟输出。服务仍可启动并通过 `/health` 报告模型不可用；实际分析任务会清晰
失败。
