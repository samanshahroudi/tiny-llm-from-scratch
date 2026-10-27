# Coverage map for N01–N04

| Source concept | Lesson and experiment |
|---|---|
| N01: next-token prediction and pipeline | 01 counting/generation; 04 complete model |
| N01: knowledge, context, memory, and hallucination | 01 memory and factuality sections; bigram limits are explicit |
| N01: context, cost rate, and latency | 01 budget example; 06 real API timing |
| N01: base/SFT/preference, temperature, and greedy decoding | 01 and 04 generation; 07 SFT path |
| N01: model role and deterministic tools | 01 deterministic Python calculation; 07 application boundary |
| N02: BPE, Unicode, whitespace, and numbers | 02; optional real-tokenizer measurement in 07 |
| N02: token/text embeddings and metrics | 02 lookup and cosine/dot/L2; optional multilingual model in 07 |
| N02: language, negation, numbers, truncation, and index | 02 limitations; optional real similarity in 07 |
| N03: Q/K/V, scaling, masking, and multi-head attention | 03 numerical experiments; 04 classes |
| N03: position/RoPE, residuals, LayerNorm, and MLP | 03 and 04; RoPE is explained but not implemented |
| N03: transformer families | 03 encoder/decoder/encoder–decoder table |
| N03: cache, GQA/MQA, O(T²), and FlashAttention | 03 single-head cache and memory calculation; main model has no cache |
| N04: corpus, splits, shifting, and shapes | 05 dataset and batch checks; 04 shapes |
| N04: cross-entropy, perplexity, and train/eval | 05 hand calculation and complete loop |
| N04: curves, checkpoints, and controlled experiments | 05 two learning rates, best/last, and round-trip |
| N04: parameters/data/compute and limits of real generation | 04 parameter count; 05/07 capacity and next steps |
| Extension: deployment and publishing | 06 FastAPI, Docker, GitHub, and Docker Space |

Optional online sections are disabled by default; a default run downloads and evaluates nothing from them. Full RAG, SFT, and LoRA training belongs to the next stage; this course defines their integration contracts and limitations.

Lesson 00 explains the required Python basics, tensors, axes, data types, and learning one weight. Later lessons introduce each new term before using it, then provide a small example, commented code, output interpretation, and answered exercise.

Lesson source is kept in `docs/*.lesson.md`; notebooks are generated from those sources. The original N04 architecture in `Labs/snippets/n04_tiny_gpt.py` inspired this project. The standalone version adds a smaller width, documented validation, separate RNGs, train-only vocabulary with `<unk>`, best-checkpoint selection, behavior tests, and service lifecycle checks.
