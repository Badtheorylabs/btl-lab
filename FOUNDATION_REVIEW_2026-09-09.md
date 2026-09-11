# Foundations for the BTL stack

This is an initial repository/documentation and paper-abstract review dated 2026-09-09. Twenty-three repository READMEs and GitHub metadata records are saved in `source-review-2026-09-09/`; Miles and the papers were additionally inspected through their primary web pages. These snapshots are discovery evidence, not pinned builds. No framework was installed or benchmarked, and paper methods have not yet been fully reproduced. Prime-RL's orchestrator received a source-level review earlier in this task; other frameworks have not received an equivalent audit.

BTL remains the lab. Tinfield is the confirmed model family. The data/environment system needs a new name. **Fieldwork** is the leading suggestion: it describes collecting experience through tasks, tools and verification. **Seedbed** and **Groundwork** are alternatives. These names are suggestions, not selected brands or package-name/trademark clearances. Existing Forge paths remain historical implementation locations.

## Recommendation

Choose foundations per layer and measured workload. Do not adopt Prime Intellect's entire stack by default. Start with one training engine and one inference backend after the compatibility comparison; supporting every backend immediately would recreate fragmentation.

Prime-RL is now the user-preferred RL foundation, pending matched validation. For large MoE post-training, retain slime/Miles and NeMo RL for model-specific code-level assessment alongside Prime-RL. Include AReaL when asynchronous scheduling is the main research variable. Keep verl as a broad comparison baseline and SkyRL as an integration candidate because existing BTL work already targets it. For general pretraining, inspect TorchTitan and Megatron Core separately: an RL orchestrator is not the entire model-training foundation.

These priorities are an engineering judgment based on documented architecture and our workload, not a speed ranking.

## Training candidates

| Candidate | Why inspect it | Main acceptance question |
|---|---|---|
| [slime](https://github.com/THUDM/slime) | Focused Megatron + SGLang path; direct upstream configuration and custom rollout generation | Does the exact Tinfield parent architecture, adapter method and context work on our intended small topology? |
| [Miles](https://github.com/radixark/miles) | Slime-derived post-training framework emphasizing production precision, stability and observability | Which changes are useful over slime, and what additional maintenance does its fork impose? |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL) | NVIDIA post-training stack with distributed backends and published model recipes | Is a recipe supported on our hardware count, or only a much larger reference cluster? |
| [AReaL](https://github.com/areal-project/AReaL) | Async training and agent integration with published staleness-control work | Does better overlap preserve quality at equal total spend on long tool-use tasks? |
| [verl](https://github.com/verl-project/verl) | Flexible RL dataflow, FSDP/Megatron training and vLLM/SGLang integration | What resource placement and complexity does the actual workload require? |
| [SkyRL](https://github.com/NovaSky-AI/SkyRL) | Unified training library, agent layer and task gym; relevant to existing BTL integrations | Can existing integrations migrate to its current interfaces without changing task semantics? |
| [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl) | Async FSDP2/vLLM stack, Verifiers and token-preserving Renderers | Does its exact supported path beat the alternatives after setup, rollout and recovery costs? |
| [TorchTitan](https://github.com/pytorch/torchtitan) / [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Distributed model-training foundations; useful for pretraining and core execution | Which model definitions, parallelism and checkpoint formats can we maintain economically? |

No winner is established. A model-family entry in a support list does not verify a modified checkpoint, expert count, LoRA layout, backward kernel or long-context configuration.

## Kernels and inference

Build a BTL kernel package around measured operators and explicit upstream provenance. A kernel library, a kernel programming language and a serving engine solve different problems.

- [Liger](https://github.com/linkedin/Liger-Kernel): fused training operations such as normalization, activation and loss paths. Benchmark realistic backward passes and memory pressure.
- [Quack](https://github.com/Dao-AILab/quack): CuTe DSL normalization, reductions, cross entropy and GEMM implementations; its README lists Hopper/Blackwell targets and CUDA requirements. Hardware suitability is a gate.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention): exact-attention implementations. Select the implementation for the GPU generation, head shape, dtype and backward requirements.
- [FLA](https://github.com/fla-org/flash-linear-attention) and [FlashQLA](https://github.com/QwenLM/FlashQLA): linear-attention/Gated DeltaNet candidates for compatible Qwen architectures. Existing BTL source review and a benchmark already live under `research/btl5_qwen27b/internal_83_60/kernels/`. Upstream code is not an original BTL kernel.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): serving-oriented attention and other GPU kernels. Evaluate within the actual inference backend.
- [CUTLASS/CuTe DSL](https://github.com/NVIDIA/cutlass), [TileLang](https://github.com/tile-ai/tilelang) and [Helion](https://github.com/pytorch/helion): implementation tools for new operators. Choose after identifying a bottleneck; account for compilation/autotuning cost and developer maintenance.

Compare [SGLang](https://github.com/sgl-project/sglang) and [vLLM](https://github.com/vllm-project/vllm) as the main serving candidates. Keep separate profiles for interactive decode, throughput serving and training rollouts. Match model revision, precision, prompt/output distribution, concurrency and quality. Neither a headline tokens/second figure nor one backend's default settings establishes a universal winner.

## Environments and research organization

Preserve [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) and [Harbor](https://github.com/harbor-framework/harbor) compatibility rather than inventing another incompatible task format. The BTL contribution can be task quality, curriculum selection, executable verification and trajectory provenance.

For Lab, compare [MLflow](https://github.com/mlflow/mlflow) and [Aim](https://github.com/aimhubio/aim) for run tracking before building a tracking UI. The BTL project registry should carry research questions, decisions, ownership and evidence gates; an existing tracker can store run metrics and artifacts. Start with the current registry until a tracking integration is actually needed.

## Papers that constrain the research direction

This table records discovery-level readings of primary abstracts/documentation. Full-method, artifact and baseline audits are the next gate before implementation or novelty claims.

| Work | Relevant mechanism | Implication |
|---|---|---|
| [HybridFlow](https://arxiv.org/abs/2409.19256) | Hybrid control/dataflow and efficient training-generation transitions | Study resource placement and resharding before designing an orchestration abstraction. |
| [AReaL](https://arxiv.org/abs/2505.24298) | Decoupled generation/training with control of stale experience | Async overlap is established prior art; evaluate learning stability as well as utilization. |
| [RollPacker](https://www.usenix.org/conference/nsdi26/presentation/gao-wei) | Tail batching to reduce long-response stalls in synchronous RL | Async is not the only route to fewer idle GPUs. The authors report 2.03–2.56x over their veRL baseline on Qwen2.5 workloads, up to 128 H800s; this is not a comparison against today's Prime-RL or our topology. |
| [AERO](https://arxiv.org/abs/2602.14338) | Adaptive rollout generation, selective rejection and Bayesian handling of zero-advantage groups | Efficient rollout allocation is already researched. Compare estimator effects and all collection cost. |
| [Adaptive Data Scheduling](https://arxiv.org/abs/2606.22305) | Semantic clusters and samples near the policy's capability boundary | Our earlier learning-progress curriculum idea needs a direct novelty comparison. |
| [FlashAttention-3](https://arxiv.org/abs/2407.08608) | Asynchronous data/computation overlap and low precision on Hopper | Hardware-specific mechanisms need architecture-specific validation. |
| [Liger Kernel](https://arxiv.org/abs/2410.10989) | Efficient fused training kernels | Memory savings and end-to-end training throughput should be measured together. |
| [FlashInfer](https://arxiv.org/abs/2501.01005) / [SGLang](https://arxiv.org/abs/2312.07104) | Attention execution and structured/prefix-sharing serving workloads | Evaluate kernels together with scheduling and cache reuse. |

Do not multiply speedups from these papers. Their baselines, workloads and optimization targets differ, and gains can overlap.

## Decision process

1. Pin candidate commits, dependency versions, actual license files and model/task artifacts. GitHub license metadata in the discovery manifest is only preliminary; `NOASSERTION` is not a license conclusion.
2. Eliminate candidates lacking a credible path for the exact architecture, backward pass, context and intended GPU topology. Inspect source and tests before paid execution.
3. Narrow to two whole-stack candidates. Use the same model and workload. Separate engine-only comparisons with matched algorithms from end-to-end recipe comparisons whose algorithms differ.
4. Check update/save/reload and behavioral correctness before steady-state throughput. Measure startup, compilation, checkpointing, failure recovery, wasted trajectories and environment/API costs as well.
5. Choose the foundation that meets the workload and maintenance constraints. Develop the first BTL contribution at a measured bottleneck. Retain the other stack as a comparison, not a second permanent implementation to support immediately.

No paid run or public release was launched by this review. The next deliverable is a pinned compatibility audit for the selected workload, not an assumed multi-fold gain.

## Unsloth added to the shortlist

[Unsloth Core](https://github.com/unslothai/unsloth) is the current backend for BTL Adapt's efficient SFT/LoRA/QLoRA path. A bounded Qwen3.5-0.8B A100 execution and recovery profile passed; a matched non-Unsloth speed baseline and production support remain open. Keep its backend environment separate from Prime-RL initially; validate checkpoint/adapter handoff before chaining stages. See the BTL Adapt qualification receipt for the exact run and limits.
