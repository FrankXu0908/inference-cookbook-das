# Qwen-Image-2.1：2026-10 历史验证记录

> 本页保留 f89209aa84a4 及当时依赖上的历史结果。当前单卡建议见 [Qwen-Image-2.1 部署](qwen-image-2.1.md)，本页命令和质量分数不代表新主仓＋正式插件的验证结果。

## 模型简介

[Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) 支持文生图、图像编辑和 RGBA 输出。本页验证 **BW1000、1024×1024、40 步纯文生图**，采用 SGLang 原生 pipeline。

**单卡推荐 BF16、Torch SDPA、权重常驻、完整帧 VAE 解码、batch=1。** 本次合批的吞吐收益很小，平均请求延迟明显增加；VAE tiling 的单卡对照也更慢。这是本页已测组合中的部署建议，不代表所有后端、分辨率和硬件的全局最优。

本文按本次 **SGLang 0.5.22** 的版本定位整理。编辑和透明背景任务不在本次性能验证范围。

## 模型列表

| 模型权重 | 精度 | SGLang 版本 | 硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | BF16 | 0.5.22 | BW1000，64 GiB/卡 | 1 | Online | [单卡低延迟](#单卡低延迟) |
| Qwen/Qwen-Image-2.1 | BF16 | 0.5.22 | BW1000，64 GiB/卡 | 4 | Online | [四卡低延迟](#四卡低延迟) |

## 环境准备

性能数据采集于 2026-10-21 至 2026-10-22，质量评分于 2026-10-23 完成，节点 a14r4n17。运行时设备名为 `BW`，板卡型号由现场确认为 BW1000。每卡 65520 MiB，架构 gfx936、80 CU；管理工具报告卡间 HSW 互联。

| 项目 | 验证环境 |
| --- | --- |
| 宿主机 / 容器 | Sugon OS 8.9 / Ubuntu 22.04.5，x86_64 |
| 驱动 / HIP | 6.3.31-V1.5.0a / 6.3.26113 |
| Python | 3.10.12 |
| PyTorch 包版本 | 2.11.0+dtk2604.2610161756.g18ae4f |
| Transformers / tokenizers | 5.17.0 / 0.23.2 |
| Diffusers | 0.37.0；使用 SGLang 原生实现 |
| 本次 SGLang 版本定位 | 0.5.22 |
| 实测源码基线 | f89209aa84a4fd0619d0662f6adf6980740efc78，加下述参数透传修复 |

### 容器

本次框架版本定位为 **0.5.22**。以下历史基础镜像仅用于提供已验证的 DTK、PyTorch 和算子依赖，容器内再 editable 安装下述源码；不以基础镜像的 `0.5.18` 标签作为本次框架版本。需要对应镜像仓库的访问权限，本文不构造未验证的 0.5.22 镜像标签。

```bash
docker pull 42.228.13.241:5000/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10-sf_b075-20261020-0337

docker run --rm -it --name qwen-image21-bw1000 \
  --network host --ipc host --privileged --shm-size 64g \
  --ulimit memlock=-1:-1 --ulimit stack=-1:-1 \
  --device /dev/kfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /data:/data --workdir /workspace \
  42.228.13.241:5000/jenkins/model_test_env/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10-sf_b075-20261020-0337 bash
```

现场镜像 ID 为 `sha256:d225cd64341976612f5ba3d2d4b9b18d8fdd744ba23476cfed968e26912b883c`。标签中的 `0.5.18` 不代表随后安装的实际源码版本。

### 安装测试源码

在容器内执行：

```bash
git clone https://github.com/HYGON-AI/sglang-das.git
cd sglang-das
git checkout f89209aa84a4fd0619d0662f6adf6980740efc78
```

该提交的 transformer loader 已传入 `streaming_state_dict_load`，中间封装缺少接收和透传。本次测试使用以下两行修复；复现此固定提交时也需要应用：

```bash
git apply - <<'PATCH'
diff --git a/python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py b/python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py
index 7901846cb0..1b43fa32ca 100644
--- a/python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py
+++ b/python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py
@@ -722,6 +722,7 @@ class WeightOverrideComponentLoader(ComponentLoader):
         *,
         component_starts_on_cpu: bool,
         weight_load_plan: WeightLoadPlan | None = None,
+        streaming_state_dict_load: bool = False,
         checkpoint_key_filter: Callable[[str], bool] | None = None,
         weights_iterator: Iterator[tuple[str, torch.Tensor]] | None = None,
     ) -> nn.Module:
@@ -740,6 +741,7 @@ class WeightOverrideComponentLoader(ComponentLoader):
             reduce_dtype=torch.float32,
             strict=False,
             weight_load_plan=weight_load_plan,
+            streaming_state_dict_load=streaming_state_dict_load,
             checkpoint_key_filter=checkpoint_key_filter,
             weights_iterator=weights_iterator,
         )
PATCH

SGLANG_BUILD_RUST_EXTS=none python3 -m pip install --no-build-isolation --no-deps -e ./python
python3 -m pip install --no-deps transformers==5.17.0 tokenizers==0.23.2
```

保留镜像中的厂商 PyTorch、Triton 和算子包，不要以通用 PyPI PyTorch 替换。本页记录已运行的依赖组合，不承诺其他版本兼容。

## 启动命令（SGLang 0.5.22）

### 单卡低延迟

选择空闲设备；以下选择物理设备 0，进程内映射为逻辑设备 0。现场单卡实验使用空闲设备 5。服务采用默认端口 30000。

```bash
HIP_VISIBLE_DEVICES=0 sglang serve \
  --model-path Qwen/Qwen-Image-2.1 \
  --model-id Qwen-Image-2.1 \
  --num-gpus 1 \
  --attention-backend torch_sdpa \
  --performance-mode speed \
  --batching-max-size 1 \
  --vae-tiling false \
  --host 0.0.0.0
```

官方模型 ID 需要可访问的模型源。本次性能测试读取本地完整权重；离线部署时替换 `--model-path` 为本地目录，保留 `--model-id Qwen-Image-2.1`。权重目录包含 `model_index.json`、`processor`、`text_encoder`、`transformer`、`vae` 和 `scheduler`。

speed 模式在本次配置下采用 eager、BF16 DiT/VAE；没有量化、torch.compile、FSDP 或 CPU offload。单卡不需要 Ulysses 拆分或多卡 VAE 空间分片。

### 四卡低延迟

需要降低单请求延迟时，使用 Ulysses 4 卡并保留已有小幅收益的 VAE 空间分片：

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 sglang serve \
  --model-path Qwen/Qwen-Image-2.1 \
  --model-id Qwen-Image-2.1 \
  --num-gpus 4 --sp-degree 4 --ulysses-degree 4 \
  --ring-degree 1 --tp-size 1 \
  --attention-backend torch_sdpa \
  --performance-mode speed \
  --batching-max-size 1 \
  --vae-tiling false \
  --vae-config.use-parallel-decode true \
  --vae-config.parallel-decode-mode spatial_shard \
  --host 0.0.0.0
```

空间分片按高度将 VAE 解码分到多卡，再收集完整图像；本次没有启用 tiling。单卡不会获得这种多卡收益。

## API 调用

```bash
curl -fsS http://localhost:30000/health

curl -fsS http://localhost:30000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen-Image-2.1",
    "prompt": "一只红色陶瓷茶壶放在木桌上，窗边自然光，写实摄影，细节清晰。",
    "size": "1024x1024",
    "n": 1,
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "seed": 42,
    "response_format": "b64_json"
  }' > response.json

python3 - <<'PY'
import base64
import json
from pathlib import Path

response = json.loads(Path("response.json").read_text())
for index, item in enumerate(response["data"]):
    Path(f"output-{index}.png").write_bytes(base64.b64decode(item["b64_json"]))
PY
```

返回 1024×1024 RGBA PNG。本页目视验证了红色茶壶、木桌和窗边光照，没有将其推广为编辑、透明背景或完整质量验证。

## 性能参考

固定上述提示词、1024×1024、40 步、CFG 1、seed 42，每请求一张图。服务和请求预热，条件缓存开启，无 profiler。端到端延迟包含 HTTP、排队、模型流水线、PNG 编码和响应读取，不是纯 GPU 时间。

### 单卡合批

2026-10-22 的每组客户端并发等于最大批量，均未超过批量上限。服务默认预热后，再执行一个完整单请求和一轮对应并发的预热。

| VAE | 最大批量 / 并发 | 实际正式批量 | 成功请求 | 平均请求延迟 | 吞吐 | 相对完整帧 batch=1 吞吐变化 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 完整帧 | 1 / 1 | 1 | 5/5 | 13.505 s | 0.0740 张/s | +0.00% |
| 完整帧 | 2 / 2 | 2 | 8/8 | 27.348 s | 0.0731 张/s | -1.21% |
| 完整帧 | 4 / 4 | 1、3、4 | 8/8 | 47.205 s | 0.0768 张/s | +3.75% |
| tiling | 1 / 1 | 1 | 5/5 | 14.167 s | 0.0706 张/s | -4.67% |

单卡以 batch=1 为部署默认。batch=2 没有吞吐收益；batch=4 的小幅收益伴随明显延迟增长。tiling、batch=1 也更慢，64 GiB 单卡能完成当前完整帧负载时，不为提速开启 tiling。

本页单卡推荐限于表内已复核的配置，没有穷举所有批量、后端或分辨率，也没有据此宣称更大批量不可运行。

### 多卡与 VAE 对照

2026-10-21，默认完整帧解码、最大批量 1、单并发，每组 5 个正式请求：

| 卡数 / Ulysses degree | 平均端到端延迟 | 相对单卡加速 |
| --- | ---: | ---: |
| 1 | 13.496 s | 1.00× |
| 2 | 10.568 s | 1.28× |
| 4 | 6.704 s | 2.01× |

多卡扩展低于线性。多卡保留 encoder auto folding，不能将整条流水线的差异单独归因于 attention 通信。

开启 VAE spatial_shard 后，双卡单并发从 10.579 s 降至 10.487 s，四卡从 6.700 s 降至 6.524 s，端到端改善约 0.9% 和 2.6%。阶段日志未显式同步 GPU，不能称为 VAE 独占 GPU 时间的加速比例。

### 四卡动态合批

四卡保留 VAE spatial_shard，等待窗口 20 ms，每档 16 个正式请求：

| 最大批量 / 并发 | 正式请求实际批量 | 平均请求延迟 | 吞吐 |
| --- | --- | ---: | ---: |
| 2 / 2 | 2 | 12.540 s | 0.1594 张/s |
| 4 / 4 | 2 | 23.318 s | 0.1608 张/s |
| 8 / 8 | 3、5 | 41.527 s | 0.1708 张/s |

上限不保证填满。等待窗口增加到 1000 ms 仍未稳定填满，吞吐也未改善；没有据此测得实际 batch=8 的纯模型性能。不同并发或 VAE 模式下的数字不能作为严格的单因素 A/B。

## 图像质量评测

使用 [Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) 的完整 1000 条中文提示词，每条生成一张图片。生成配置为原生 Ulysses 4 卡、VAE spatial_shard、BF16、Torch SDPA、1024×1024、40 步、CFG 1、seed 42、单并发。1000 张图片全部生成并完成评分，图片校验和评分输出完整性校验通过。

评分模型使用官方 [Qwen/Qwen-Image-Bench](https://modelscope.cn/models/Qwen/Qwen-Image-Bench) 专用 Q-Judger BF16 权重，由 Qwen3.6-27B 微调。评分运行于独立环境，采用 ms-swift 4.2.1、Transformers 5.8.1，保留官方 checklist、temperature=0、top_k=1、top_p=1、repetition_penalty=1.05、seed=42、thinking=True 和 max_new_tokens=4096；评分批量上限为 12。前 4 张评分批量为 4，随后上限 24 遇到显存不足，降到 12 续跑。最终成绩均来自官方 ms-swift 后端；提速候选的输出单独保留。

| 维度 | 分数 / 100 |
| --- | ---: |
| 图像质量 Quality | 53.83 |
| 美学 Aesthetics | 53.61 |
| 提示词对齐 Alignment | 54.27 |
| 现实一致性 Real-world Fidelity | 49.41 |
| 创意生成 Creative Generation | 52.34 |
| 综合分 Overall | **53.22** |

按官方规则逐级聚合，综合分先对每张图片的适用一级维度求平均，再对 1000 张求平均。第 920 条的创意生成维度输出全部为 N/A，依官方规则排除 N/A，保留原始输出，没有人工补分或重评挑分。该结果评价本页生成配置的图像质量；没有同配置参考设备输出，尚不能据此判定跨硬件的数值精度损失。

## 合批复现与日志检查

复现合批时，将单卡命令的最大批量改为待测值，并增加：

```bash
--batching-delay-ms 20 \
--enable-batching-metrics
```

压测客户端随本 cookbook 仓库提供，与 `sglang-das` 源码目录分开。将本仓库挂载或克隆到容器后，在 **inference-cookbook-das 仓库根目录**执行以下命令复现 batch=2，不改变默认 batch=1 的建议：

```bash
python3 docs/model-deployment/sglang/scripts/qwen-image21-benchmark.py \
  --url http://127.0.0.1:30000 \
  --max-batch-size 2 --concurrency 2 \
  --requests 8 --output qwen-image21-batch2
```

客户端拒绝并发超过传入的最大批量，保存请求、逐次响应元信息、汇总和 PNG 样例。传入上限应与实际服务一致。客户端默认两轮对应并发的预热，与上表当次测试的预热轮数不同。

核对日志中的实际批量：

```text
Dynamic batch dispatch: requests=2, outputs=2/2, user_max_outputs=2
Processed dynamic batch of 2/2 request(s) with max_delay=20.00ms
```

只看到配置上限不能证明执行了该批量。当前版本没有按实时空闲显存自动选择批量；没有 admission 规则时使用用户上限。纯文生图合批不能直接推广到编辑请求。

## 关键参数与验证边界

- `--attention-backend torch_sdpa`：已测的精确注意力后端，未穷举其他后端。
- `--performance-mode speed`：本次保持 eager 和权重常驻，未开启 compile。
- `--vae-tiling false`：单卡完整帧解码。tiling 可降低解码工作集，但可能改变边界像素，不等于多卡 VAE 并行。
- `--vae-config.parallel-decode-mode spatial_shard`：多卡解码组中的空间分片；本页多卡建议保留，单卡无此多卡收益。
- `--batching-max-size`：请求合并上限，实际批量还取决于到达节奏和兼容性。
- `ulysses_async`：本次原生 QwenImage21 链路没有有效的 cache-dit 异步 patch，不提供假定已生效的开启命令。

不同批量和并行度会改变浮点计算顺序，即使固定 seed，输出也不保证逐像素相同。本页没有验证长期稳定性、所有采样参数或其他分辨率的最优配置。
