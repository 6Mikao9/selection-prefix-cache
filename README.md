# Selection Prefix Cache

一个可运行的研究原型：KV 前缀缓存被淘汰后，保留压缩的稀疏选择记录；重复 prefill 时，分块预取并提前解压选择索引，让当前计算与后续准备重叠。

**A small research prototype for compressed selection replay with bounded CUDA buffers.** Storage, prefetch/decode and head/tail splitting are tested. Experimental serving adapters are included, but native model replay and end-to-end speedup remain unverified.

## 做了什么

- 每个 chunk 保存二维逻辑块 ID（每行固定 k），支持 `raw_u16` 和共享字典 + 逐行位图；自动模式在位图膨胀时退回 raw。
- `data.bin` 保存连续编码包，`manifest.json` 保存偏移、长度、校验和及缓存身份；支持随机读取，不把整个文件展开进内存。
- 既可从磁盘文件逐块读，也可将压缩包留在 DRAM。
- CUDA 路径预分配有限的 pinned 输入、GPU 压缩输入和 int32 输出缓冲。独立 stream 提前搬运、解码；消费者用完后复用缓冲。
- 单个后台读取线程和有限预取窗口避免全量载入。事件约束数据就绪与缓冲复用，保留串行对照路径。
- 支持前段重算、仅后段缓存：静态切点对齐 chunk，前段调用真实重算回调，同时准备后段；缓存文件不保存前段选集。

流程为：`SSD/DRAM 压缩记录 → 有界 pinned staging → GPU 提前解码 → 当前 stream 消费 → 缓冲复用`。

使用相同完整因果前缀的缓存记录时，解码后续选择不依赖本次尚未计算出来的 Q/K。这使得按实际层/chunk 消费顺序提前准备成为可能。准备吞吐仍需跟得上消费，首段也可能需要等待。

## 运行

Python 3.10+；CPU 路径需要 NumPy。CUDA 路径另外需要兼容的 PyTorch、Triton 和 NVIDIA GPU。脚本不会安装或升级环境；建议使用已有环境。

```bash
# CPU codec / file-store demo。输出目录必须不存在。
python demo.py --device cpu --directory ./demo-cache

# CPU 测试；没有 CUDA/Triton 时 GPU 测试会跳过。
python -m unittest discover -s tests -v

# 仅选择已确认空闲的 GPU，按机器情况替换 0。
CUDA_VISIBLE_DEVICES=0 python demo.py --device cuda \
  --directory ./demo-cache-gpu --chunks 12 --rows 256 --k 512 \
  --universe 8192 --compute-size 1024 --repeats 3 --output results.json
```

Demo 生成合成选择记录。CUDA 对照使用合成矩阵乘法作为消费者，比较串行与预取路径；它不是 attention、QSA 或真实 prefill。反复读取磁盘文件可能命中操作系统 page cache，因此其时间不能称为真实冷 SSD 性能。首次编译、预热与稳态测量的边界以 demo 输出和实现为准。

`--source memory` 使用压缩包驻留 DRAM 的路径；默认 `disk` 按偏移逐块读文件。`--buffers` 控制循环缓冲数，`--compute-iters` 控制合成消费者每块执行的矩阵乘法次数。Demo 在完整索引核验及一次消费者预热之后采样；每次采样包括整个流的读取、传输、解码、计算和首尾开销。

## 已运行的烟雾验证

2026-10-31，在 A100-SXM4-80GB、PyTorch 2.8.0+cu128 上，17 项 CPU/CUDA 测试全部通过。覆盖多次缓冲复用、混合 k/行数、跨多个 tile 的位图解码、ID 65535、非默认消费 stream、缓存身份/损坏检查及异常关闭。

下表来自一次合成运行：16 块 × 256 行 × 512 个 ID，候选范围 8192，2 个缓冲；每块消费 4 次 2048×2048 BF16 矩阵乘法。单位是**整个 16 块流的 wall time 中位数（ms）**，每模式 5 次重复。

| 来源 | raw 串行 | raw 预取 | 压缩串行 | 压缩预取 |
|---|---:|---:|---:|---:|
| DRAM | 12.249 | 9.281 | 12.574 | 9.231 |
| 文件读取（已预热） | 18.617 | 9.653 | 14.382 | 8.423 |

两份展开输出缓冲共 1 MiB。合成数据有效载荷由 4,194,304 B 降至 1,526,144 B；16 块中 11 块使用位图、5 块回退 raw，另计约 3.2 KB manifest。数据刻意混合高相关和独立选集，**这个容量比不是任何真实模型的压缩率**。

这表明提前准备路径可运行，并在本次合成负载下缩短整段 wall time；不是对真实模型的加速证明。DRAM 的两种预取时间接近，不据微小差异判断胜负。机器并非独占、模式顺序固定、文件已预热；稳健性能结论仍需真实消费者和更多受控测量。原始样本、配置和源码哈希见 [benchmarks/a100_smoke.json](benchmarks/a100_smoke.json)。

## 最小接口

```python
import numpy as np
from selection_cache.store import CacheIdentity, DiskStore, encode_chunk, write_store

identity = CacheIdentity(
    model_fingerprint="weights-and-adapter-version",
    prefix_digest="hash-of-the-complete-causal-prefix",
    selector_fingerprint="selector-config-and-numerics-version",
    position_mask_fingerprint="positions-rope-and-attention-mask",
)
indices = np.array([[8, 2, 6], [2, 6, 9]], dtype=np.int32)
write_store("new-cache", identity, [encode_chunk(indices)])
store = DiskStore("new-cache", expected_identity=identity)

# 在 CUDA 环境中：
from selection_cache.cuda import CudaReplay

with CudaReplay(store, buffers=2, prefetch=True) as replay:
    def consume(block_ids, ordinal):
        # 在当前 CUDA stream 上消费 block_ids；例如调用已有稀疏 attention。
        # 这里是借用的缓冲，回调返回后不可留存引用用于后续读取。
        # 需要留存结果时，在当前 stream 上 clone()。
        pass
    stats = replay.run(consume)
```

调用方负责生成正确的身份指纹；这些字符串不是自动验证模型语义的机制。完整前缀、权重/adapter、位置/RoPE、mask、selector 算法或数值路径变化时，应使对应缓存失效。新 suffix 仍需要自己的 selector 计算。

`run()` 返回整个调用的 wall time、主机读取/缓冲复用等待，以及各类缓冲字节数。首轮调用可能包含 JIT 编译，做性能比较前需要预热。主机等待统计不是 GPU stall 时间，不能直接相加为端到端收益。

## 前段重算、后段缓存

`SplitPlan` 指定总 chunk 数及前段重算数，`from_fraction()` 向下取整到 chunk 边界。持久化时只写后段包，不能仅把前段留在文件里再跳过读取。缓存身份应包含切点、chunk 布局及模型/前缀指纹。

```python
from selection_cache import SplitPlan

plan = SplitPlan(total_chunks=16, recompute_chunks=4)
# tail_store 只含原始 chunk 4..15 的十二个包，按实际消费顺序排列。
# recompute(global_ordinal) 在当前 CUDA stream 执行 selector，返回 int32 索引。
# consume(indices, global_ordinal) 对重算和重放结果执行相同的下游计算。
with CudaReplay(tail_store, buffers=2, prefetch=True) as replay:
    stats = replay.run_hybrid(consume, recompute, plan)
```

后段读取在前段开始前提交；前段工作入队后，主机非阻塞检查已读好的后段包并提前解码。消费者顺序仍为 0..N-1，事件保护缓冲复用。全重算允许空 store，选择输入/输出缓冲为零；全缓存与原 `run()` 兼容。当前是固定切点，不包含预测到达时间、动态取消或自动选择最优 H。多层接入必须由调用方明确层与 query 区间，不能把层优先排列中的前几个 chunk 自动当作所有层的前段 token。

复现实验：

```bash
# 默认切点 0、1/16、1/8、1/4、1/2、1；同时比较 DRAM 和已预热文件。
CUDA_VISIBLE_DEVICES=0 python split_demo.py \
  --directory split-cache-small --output results-split-small.json

# 更大的候选集合，仍是合成 selector 微基准。
CUDA_VISIBLE_DEVICES=0 python split_demo.py \
  --directory split-cache-long --output results-split-long.json \
  --chunks 16 --rows 128 --k 512 --universe 65536 --head-dim 128 --repeats 7
```

这里真正运行 BF16 QK、top-k 和规范排序；没有用 sleep 模拟重算。候选数逐 chunk 增长，同一 chunk 内各行共享候选范围，不是原生 QSA 的逐行因果计算。各模式预热后按固定种子的随机顺序交错采样；计时包含前段 selector、后段准备、所有消费和首尾开销。只在验证阶段保留全量参考索引，缓存产物本身只保存尾段。详见 [切分结果](benchmarks/split_results.md)。

## 格式与显存

ID 范围为 `0..65535`；65535 是合法 ID，不作为 sentinel。拒绝浮点、重复 ID、负值、越界值以及空行/空 chunk。每个 chunk 的 k 固定，不同 chunk 可独立编码。集合会排序，因此格式保证选集无损，**不保证原始 top-k 顺序或 attention 输出逐 bit 相同**。

若每个缓冲覆盖一层的 C 个 query，每个 query 512 个 int32 块 ID，则两个输出缓冲为 `2 × C × 512 × 4` 字节。C=2048 时为 8 MiB，另外计入压缩输入、pinned 内存和模型工作区。若消费内核需要展开为 2048 个 token 位置/query，展开缓冲还会变大；本原型保留块 ID，不完成这个适配。

主 KV、Q/K/V 投影、attention、MLP/MoE、GDN 状态重建仍由模型执行。用完即复用的是**选择索引临时缓冲**，不是主 KV。压缩主副本仍可按独立的缓存策略保留。

## 边界

`integrations/` 包含实验性的 vLLM DSA 和 SGLang SALA 适配脚本，以及已执行的 Qwen3-4B dense 质量基线。原生稀疏模型重放尚未通过：DSA 遇到后端依赖不兼容，SALA 遇到设备 CUDA 故障。它们不是生产插件，也没有实现 prefix lookup/淘汰和生产请求调度。详见[框架执行结果](benchmarks/framework_results.md)与[接入说明](integrations/README.md)。

缓存文件仅面向本程序写出的可信记录。写入和内存存储构造时会验证编码内容，磁盘读取检查包长度与 SHA-256；校验和用于发现损坏，不提供来源认证，也不使任意第三方缓存文件可信。

压缩减少保存或传输字节，不自动减少请求延迟。GPU 解码可能与主计算争用资源，SSD/DRAM 带宽不足也会造成等待。应在相同缓冲、预取和计时策略下比较 raw 与压缩路径，再测真实模型的端到端收益。

新增研究方向：[利用已知选集预排 KV 连续槽位](benchmarks/kv_layout_design.md)，让 attention 更规整地从 HBM 分 tile 搬入 shared memory，并考虑与 HiSparse 换入过程融合。当前只有设计，尚未实现或测得布局收益。
