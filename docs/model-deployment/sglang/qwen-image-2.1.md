# Qwen-Image-2.1 on SGLang Diffusion

## 模型简介

[Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) 支持文生图、图像编辑和 RGBA 输出。本页记录 Hygon 单卡纯文生图的部署建议，已完成 720p、1080p、2K、4K 四个横图档位对照，40 步、BF16；此前 1024×1024 和 1280×768 数据保留为历史小尺寸参考。

**部署建议：通用入口默认使用插件 eager；固定4K持续生成使用提前预热的VAE编译实例。** 两种配置均保持DiT插件eager、Torch SDPA、batch=1、完整帧VAE，关闭BCG和CPU offload。通用入口关闭全局compile；VAE编译实例通过下文独立实验入口仅编译decode，须先完成目标尺寸的真实请求预热，再接入流量。新尺寸先离线验证，避免在线请求触发长时间编译。

本轮720p至2K两模式延迟接近，VAE编译主要降低allocated/reserved。4K完整HTTP均值255.09→197.53秒，观测下降22.56%；两模式使用不同物理卡，此数值包含卡间差异。编译实例启动显存应按包含预热的峰值准备；进程重启后的持久缓存命中尚待验证。

本文为 2026-11-03 的 cookbook 草稿。默认命令依据已运行服务整理；插件交付与全新环境安装仍需补齐。编辑、透明背景和其他分辨率不在本轮验证范围。

## 模型列表

| 模型权重 | 精度 | SGLang 源码 | 硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | BF16 | 594e1ae94c61 | Hygon gfx936，65520 MiB/卡 | 1 | Online | [单卡插件 eager](#单卡插件-eager) |

## 验证环境

本轮在 a14r4n14 的 `qwen21_standard_20261027_n14` 容器验证：VAE编译组使用物理GPU0，普通eager组使用物理GPU1。当前节点正式板卡型号待补充，不由 gfx936 或显存容量推定；历史 BW1000 数据单独保留。

| 项目 | 本轮环境 |
| --- | --- |
| 主仓 | HYGON-AI/sglang-das，594e1ae94c6103dabed4152a4ebbcca6fb049aeb |
| SGLang 包标记 | 0.5.22+g594e1ae94c61，项目构建标记，非官方 release tag |
| 正式插件 | sglang-hygon-plugin，6985a7c5852c90554ca5bed158f8a35f82eadef4，六个 hook，含 SiLU-mul |
| 实际安装 | 主仓与正式插件均为 editable；不使用 dev worktree 的未合并优化 |
| Python / PyTorch | Python 3.10 / 厂商 PyTorch 2.11.0（g18ae4f） |
| HIP / 架构 | 6.3.26113 / gfx936 |
| 可用显存 | 65520 MiB/卡 |

使用已准备好的 HCU 环境，保留厂商 PyTorch、Triton 和算子依赖。当前主仓在 Python 3.10 上按仓库 `pyproject_hcu.toml` 安装；不要直接套用历史默认 pyproject 的安装步骤。正式插件部署前需提供对应提交源码并安装，目前不虚构公开下载地址或镜像标签。

启动前确认实际导入位置及插件 entry point：

```bash
python3 - <<'PYTHON'
import importlib.metadata as metadata
import sglang
import sglang_hygon.plugin
print(metadata.version("sglang"), sglang.__file__)
print(sglang_hygon.plugin.__file__)
print([(e.name, e.value) for e in metadata.entry_points(
    group="sglang.multimodal_gen.plugins")])
PYTHON
```

应能看到 `hygon_qwen_image21 = sglang_hygon.plugin:register`。仅 import 成功不代表 hook 已实际执行，仍需检查启动和真实请求日志。

## 启动命令

### 单卡插件 eager

在已准备好的 HCU 容器内执行，选择空闲物理 GPU。以下使用默认 HTTP 端口 30000，本轮普通eager实测服务为 30026；更换端口不构成新性能测量。修改本地权重路径为实际完整模型目录。

```bash
source /opt/dtk/env.sh
export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"
export HIP_VISIBLE_DEVICES=0
export SGLANG_PLUGINS=hygon_qwen_image21

sglang serve \
  --model-path /mnt/code/models/Qwen-Image-2.1 \
  --model-id Qwen-Image-2.1 \
  --num-gpus 1 --tp-size 1 \
  --sp-degree 1 --ulysses-degree 1 --ring-degree 1 \
  --attention-backend torch_sdpa \
  --performance-mode speed \
  --batching-max-size 1 \
  --vae-tiling false \
  --enable-torch-compile false \
  --enable-breakable-cuda-graph false \
  --regional-compile false \
  --offload-during-compile false \
  --warmup-mode off \
  --host 0.0.0.0 --port 30000 \
  --output-path outputs
```

`speed` 在本轮配置下使用 BF16、权重常驻。服务就绪后，由客户端用固定 prompt 做 3 次预热，再测 5 次正式请求；启动命令本身不发送预热请求。首次请求可能包含插件 JIT 和库初始化，不能算作稳态延迟。

## API 调用

```bash
curl -fsS http://127.0.0.1:30000/health

curl -fsS http://127.0.0.1:30000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen-Image-2.1",
    "prompt": "一只红色陶瓷茶壶放在木桌上，窗边自然光，写实摄影，细节清晰。",
    "size": "1280x736",
    "n": 1,
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "seed": 42,
    "response_format": "b64_json"
  }' > response.json

python3 - <<'PYTHON'
import base64, json
from pathlib import Path
response = json.loads(Path("response.json").read_text())
Path("output.png").write_bytes(base64.b64decode(response["data"][0]["b64_json"]))
print("inference_time_s:", response.get("inference_time_s"))
PYTHON
```

其他测试档位使用 `1920x1088`、`2560x1440`、`3840x2176`；这些值满足32倍数约束。接口返回 PNG；固定 seed 不保证不同编译或并行模式逐像素一致。

## 当前分辨率口径（2026-11-03）

当前实现要求宽高均为32的倍数。用户选择直接测试对齐后的完整原图，不裁剪、不缩放：

| 档位 | 常见标准尺寸 | 本页实际测试尺寸 |
| --- | --- | --- |
| 720p 档 | 1280×720 | **1280×736** |
| 1080p 档 | 1920×1080 | **1920×1088** |
| 2K / 1440p 档 | 2560×1440 | **2560×1440** |
| 4K 档 | 3840×2160 | **3840×2176** |

2K采用常见显示器1440p口径。表内对齐尺寸不都严格为16:9，性能和显存按实际生成像素计算；720p/1080p/4K仅表示接近常见分辨率的档位，不能写成标准尺寸原生出图。若需要标准尺寸交付，可以另做边缘裁剪，裁剪不会省去模型已经执行的计算。

四档结果见下表。所有建议仅覆盖本次固定prompt、40步和实测运行配置。

<!-- resolution-tiers-results:start -->
## 四档性能与显存（2026-11-03）

固定本页中文prompt、seed42、40steps、CFG1、BF16、Torch SDPA、n1、batch1、TP/SP1、BCG/offload=false、完整帧VAE，普通eager四档与VAE-only 4K为3次客户端预热＋5次正式；VAE-only前三档保留此前已完成的10次预热＋5次正式。4K原始旧计划warmup标记保留，按索引0–2预热、3–7正式汇总。组内首次失败则停止，不将失败或编译成本计入正式均值。重复prompt允许编码缓存命中。不开profiler，原生perf_dump采集阶段与显存快照。

| 档位 | 实际尺寸 | 配置 | HTTP均值s | API inference均值s | forward peak allocated GiB | API peak reserved GiB | 含预热整卡采样峰值GiB | 成功/请求 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 720p | 1280x736 | 插件 eager | 9.7306 | 9.3001 | 45.647 | 53.436 | 54.155 | 8/8 |
| 720p | 1280x736 | 插件 eager＋VAE 编译 | 9.7495 | 9.2954 | 32.415 | 34.914 | 35.640 | 15/15 |
| 1080p | 1920x1088 | 插件 eager | 25.0219 | 23.8618 | 39.338 | 47.811 | 54.155 | 8/8 |
| 1080p | 1920x1088 | 插件 eager＋VAE 编译 | 25.1802 | 24.1927 | 34.883 | 38.906 | 39.629 | 15/15 |
| 2K | 2560x1440 | 插件 eager | 51.1549 | 49.1070 | 46.186 | 59.996 | 60.726 | 8/8 |
| 2K | 2560x1440 | 插件 eager＋VAE 编译 | 51.3062 | 49.5293 | 38.318 | 43.748 | 44.470 | 15/15 |
| 4K | 3840x2176 | 插件 eager | 255.0907 | 166.6377 | 55.085 | 61.012 | 61.710 | 8/8 |
| 4K | 3840x2176 | 插件 eager＋VAE 编译 | 197.5340 | 167.5699 | 48.364 | 58.781 | 63.473 | 8/8 |

HTTP包含最终输出等待、图片编码和传输；inference为服务端响应字段，当前源码在materialize/finalize输出前记录total_duration_ms，可能未包含后续GPU异步完成等待，不是同步GPU计时，HTTP减inference不能直接当作编码成本。allocated是张量分配峰值，reserved包含缓存，两者不能相加；PyTorch两列取正式请求峰值：allocated来自原生after_forward快照，reserved来自API最终高水位，allocated未另采输出落地阶段计数；整卡列覆盖本组全部请求（含预热），以sysfs每20ms采样，可能遗漏短尖峰。失败前峰值不代表完成该请求需要的容量。

VAE编译组复用GPU0已有服务和前一天生成的缓存，普通eager组依用户要求在GPU1与独立端口并行运行；各模式内部升序换形状，显存包含历史缓存影响。不同物理卡的对照可能包含卡间差异，小幅延迟变化不能全归因VAE编译。这是当前配置下的连续请求对照，不是每个尺寸独立冷启动或长期稳定性测试。
<!-- resolution-tiers-results:end -->


4K另补一次40步热请求：普通eager251.65秒，VAE编译192.77秒。随后每模式独立1步4K的原生GPU trace显示：三个大卷积逻辑shape相同，内存布局由NCHW变为NHWC，实际卷积kernel合计83.97→25.94秒，解释约58秒差距；去噪1步GPU kernel合计均约3.98秒。因此本轮4K大幅收益主要来自卷积布局与执行内核变化。已有1024与小尺寸动态图也包含channels-last卷积，布局优化早已存在；4K出现的是明显的耗时收益，并非首次改布局。补采2K的1步GPU trace显示，eager已使用implicit-GEMM等卷积内核，解码kernel合计1.256→1.011秒，仅节约约0.244秒，相对约51秒的40步请求很小；720p和1080p尚未采集对应trace。诊断数据不加入40步基准。诊断1步请求不混入40步基准；后续用当前MIOpen库的descriptor查询验证：同一288通道3×3卷积，空间2048×3640→2048×3641、元素数跨过2^31时，原快速implicit-GEMM候选被判不适用，两布局均切换到fallback；这支持当前库的大张量索引/支持范围限制解释。此为无大张量分配的候选查询，厂商solver具体源码条件仍未取得；两物理卡对照限制保留。

## VAE 单独编译与缓存

VAE-only模式保持DiT插件eager、BCG=false，只编译VAE decode（`default`）。当前CLI只有全局编译开关，实验通过独立插件复用原生解码编译函数，尚未并入正式插件；不能用 `--enable-torch-compile true` 冒充VAE-only。

在已验证的standard容器里，实验启动入口为：

```bash
MODE=plugin \
GPU_ID=0 PORT=30024 SCHEDULER_PORT=5684 MASTER_PORT=30005 \
ENABLE_TORCH_COMPILE=false \
ENABLE_BREAKABLE_CUDA_GRAPH=false \
OFFLOAD_DURING_COMPILE=false \
bash /mnt/logs/model-adaptation/qwen-image-2.1/plugin-eager-vae-compile-20261102/run.sh --inside
```

该入口默认GPU0/HTTP30024，并依赖项目日志目录下的bootstrap及本地entry point，不是独立发布的cookbook安装包。启动前核对目标卡和端口。正式插件的VAE组件开关与交付方式仍待整理，本文不虚构新CLI。

本轮720p/1080p/2K档复用了已有动态图，未出现长时间重编译；4K超过原图32位索引条件，触发新图编译，首请求843.36s，包含编译、初始化和推理，不能当作纯编译耗时或稳态延迟。

4K正式请求的forward peak allocated由55.085降至48.364GiB，API peak reserved由61.012降至58.781GiB；但包含预热的整卡采样峰值反而由61.710升至63.473GiB，不能称为所有阶段都更省显存。4K VAE编译首请求843.36秒，稳态HTTP均值197.53秒；应根据连续出图次数和启动显存余量选择。

持久化目录只保存和复用已经生成的缓存，不消除首次编译，也不保证每种形状都命中。保留 `TORCHINDUCTOR_CACHE_DIR` 与 `TRITON_CACHE_DIR`，并记录软件版本、形状和模式；本轮未验证进程重启后的命中情况。H20参考日志使用已有持久缓存，10次预热合计约4分18秒、后五次约19.9秒，缓存条件和设备均与本轮不同，不能直接比较编译速度。

[2026-11-02小尺寸结果](qwen-image-2.1-history-20261102.md) 保留原生compile、插件eager、BCG三模式和VAE-only初测，供追溯。BCG在两种旧尺寸下HTTP延迟增加约4%，本页暂不推荐；没有把该结果推广为四个新档位的BCG性能。

## 客户端预热与复测

在本 cookbook 仓库根目录执行：

```bash
python3 docs/model-deployment/sglang/scripts/qwen-image21-benchmark.py \
  --url http://127.0.0.1:30000 \
  --size 1280x736 \
  --max-batch-size 1 --concurrency 1 \
  --warmup-rounds 3 --requests 5 \
  --output qwen-image21-eager-720p-aligned
```

其他档位修改 `--size` 并使用新的输出目录。脚本保存预热、正式请求、汇总和样例 PNG。本轮原始性能由项目实验客户端采集，cookbook 客户端用于复测相同请求口径。

## 日志检查

检查启动日志包含六个正式插件 hook 的实际应用信息，来源为 `hygon_qwen_image21` / `sglang-hygon-plugin`。真实请求应进入 `QwenImage21DenoisingStage` 和 `DecodingStage`；首次 KV pack 数值准入应通过。仅设置 `SGLANG_PLUGINS` 不能证明插件已生效。

推荐命令不应出现 DiT/VAE 的 torch.compile 或 BCG 捕获日志；若出现，检查是否从其他 shell 继承了实验启动方式、插件或参数。

## 历史结果与后续补齐

[2026-10 历史验证记录](qwen-image-2.1-history-202610.md) 保留旧 f89209aa84a4 基线的多卡、合批、VAE spatial_shard 和官方千图评分，供追溯。历史 53.22 分不能当作本轮新主仓＋正式插件的质量成绩，多卡命令也未在本轮版本重新验证。

发布前补齐：正式插件源码获取与新环境安装、当前节点正式板卡型号、VAE-only 的正式组件开关与质量验证、正式插件质量评分和长期稳定性验证。本页不提供尚未测量的多卡最优命令。

本页四档原始证据在项目记录 `logs/model-adaptation/qwen-image-2.1/resolution-tiers-20261103/`；历史小尺寸记录在 `shape-plugin-eager-20261102/` 与 `plugin-eager-vae-compile-20261102/`：包含实际启动命令、runtime、逐请求JSON、图片、源码哈希和编译证据。本轮GPU0/HTTP30024保留VAE-only服务，GPU1/HTTP30026为普通eager；本文默认启动示例使用普通eager，两种入口不要混淆。
