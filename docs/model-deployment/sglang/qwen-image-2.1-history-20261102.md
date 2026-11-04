# Qwen-Image-2.1：2026-11-02 小尺寸对照

本页保留主仓594e1ae94c61、正式插件6985a7c上的1024×1024和1280×768结果。当前横图四档与推荐命令见 [部署页](qwen-image-2.1.md)。

## 2026-11-02 小尺寸性能参考

同日同版本、固定上述中文 prompt、40 步、CFG1、seed42、n1、BF16、Torch SDPA、TP/SP1、VAE tiling=false。每组 10 次客户端预热＋5 次正式串行 HTTP，无 profiler。重复 prompt 允许编码缓存命中。

HTTP 从发送请求计至完整 JSON 读取结束，包含图片编码和传输；inference 为服务端响应字段。两者不能混作同一个指标。

| 形状 | 配置 | HTTP均值 s/张 | 服务端 inference均值 s | peak reserved GiB |
| --- | --- | ---: | ---: | ---: |
| 1024×1024 | 原生 compile | 11.0463 | 10.8067 | 35.727 |
| 1024×1024 | **插件 eager（推荐）** | **10.9186** | **10.4235** | 55.873 |
| 1024×1024 | 插件 eager＋BCG | 11.3195 | 10.8241 | 58.082 |
| 1280×768 | 原生 compile | 9.6138 | 9.4003 | 35.850 |
| 1280×768 | **插件 eager（推荐）** | **9.6164** | **9.1808** | 54.373 |
| 1280×768 | 插件 eager＋BCG | 10.0068 | 9.5737 | 58.082 |

BCG 在两种形状下的 HTTP 延迟分别比插件 eager 增加 3.67% 和 4.06%，暂不推荐。已由 profile 确认真实图回放，两组输出均与插件 eager 逐像素一致；变慢的具体原因仍待同口径 trace 对照，不能只归因于 graph break。

原生 compile 为 DiT `max-autotune-no-cudagraphs`、VAE `default`，沿用旧服务 warmup=server、offload=true；插件 eager 为 warmup=off、offload=false。两者属于整套运行配置对照，不能将全部差异归因于某个算子。原生与插件输出并非逐像素一致。

显存列为 API `peak_memory_mb` 实际返回的 PyTorch peak reserved，经 1024 换算为 GiB；不是 peak allocated 或整卡占用。

### VAE 单独编译实验

DiT 保持插件 eager、BCG=false，只编译 VAE decode（`default`）。当前 CLI 只有全局编译开关，实验通过独立插件复用原生解码编译函数，尚未并入正式插件；**不能用 `--enable-torch-compile true` 冒充 VAE-only**。

1024×1024 的 HTTP 为 10.8732 s，inference 为 10.4200 s，相比插件 eager 分别只降低 0.42% 和 0.03%；peak reserved 从 55.873 降至 35.615 GiB。与 eager 图片的 RGB PSNR 为 56.64 dB，并非逐像素一致。当前更适合作为省显存候选，尚未完成完整质量评分。

1280×768 的 HTTP 为 9.6387 s，inference 为 9.1844 s，比 eager 分别增加 0.23% 和 0.04%；peak reserved 从 54.373 降至 35.572 GiB，RGB PSNR 为 57.09 dB。两种形状的速度差异均很小，主要收益为降低 reserved 显存。

首次方图请求为 407.74 s，换横图首次请求为 555.82 s，包含编译、初始化和推理；不计入正式均值。两组各 15 张输出内部哈希一致。暂不作为主推荐命令或正式组件开关发布。

### 编译缓存与启动成本

H20 参考日志当时复用已有持久化缓存，关闭服务自动 warmup，由客户端固定 prompt 预热 10 次，合计约 4分18秒，后五次约19.9秒。这些数字来自参考设备，不能替代本页 Hygon 测量。

持久化目录只能保存和复用已生成的缓存，不会消除首次编译。当前 VAE-only 实验使用独立空缓存目录，换形状也触发重编译；与 H20 的热缓存条件不同。需要编译时，应长期保留 `TORCHINDUCTOR_CACHE_DIR` 与 `TRITON_CACHE_DIR`，并记录软件版本、形状和编译模式；缓存目录存在不等于实际命中。
