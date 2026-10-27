# Interview practice using this project

Answer aloud without looking at the notes. If tensor shapes are unclear, return to the cell and print the shape.

1. **Explain the model in one minute.** Text becomes IDs; each ID looks up a trainable embedding vector and receives a positional embedding. Causal-attention and MLP blocks transform the representation. The output head produces one logit per character. During training, targets are shifted one position and cross-entropy is computed; during inference, the final logit is sampled repeatedly.
2. **Why can fluent output be wrong?** The training objective models the probability of observed tokens; it does not verify claims about the world. Grounding, deterministic tools, source checks, and evaluation must be added. Token probability is not sentence truth probability.
3. **How do an ID, token embedding, and contextual representation differ?** An ID is an integer index. A token embedding is a trainable vector looked up from that ID. Attention makes the vector context-dependent. RAG needs an embedding model trained for retrieval, not merely an average of character vectors.
4. **What does BPE do?** It merges frequent pairs of initial units and applies the learned rules in order to new input. Byte-level BPE starts from 256 byte values and can represent new text without a whole-word vocabulary. Pre-tokenization and special-token rules matter in a real tokenizer.
5. **Explain attention with shapes.** `X` has shape `B×T×C`; each head's Q/K/V has shape `B×T×(C/H)`. `QKᵀ` has shape `B×T×T`; divide by `√d`, apply the causal mask before softmax over keys, then multiply by V. Concatenation and projection return `B×T×C`.
6. **Why LayerNorm and residuals?** LayerNorm normalizes features at each position. A residual path preserves representation and gradient flow. The MLP transforms each position independently.
7. **Why is training parallel but generation stepwise?** With known targets, all positions can be evaluated in one masked forward pass. During generation, the next token does not exist yet; the next input depends on the current output.
8. **What is a KV cache?** Intermediate K/V values for previous tokens are stored so decoding does not recompute the prefix. The cache costs memory and is not user memory. The educational GPT has no cache; notebook 03 compares one head numerically. GQA/MQA reduce the number of K/V heads. FlashAttention improves memory traffic; it does not make dense attention inherently linear.
9. **Why does the training target shift?** At position `t`, `x` contains the prefix through `t` and must predict token `t+1`. Using `target=input` or removing the mask can produce a good-looking loss for the wrong task.
10. **How did you check overfitting?** Compare train/validation curves in evaluation mode, keep document IDs/hashes independent, and inspect shared templates. Falling train loss with rising validation loss is one signal; distribution shift also matters. Evaluate test only after selecting settings.
11. **What does a checkpoint contain?** Weights, configuration, vocabulary, and metadata. This project compares logits exactly before and after loading. Optimizer state is needed for a training resume; exact resumption also depends on RNGs, the scheduler, and data state.
12. **Why choose this model?** It has 114,515 parameters, two width-64 blocks, and four heads, so it is readable and runs on CPU. It can be exposed through an API and packaged with Docker, but its quality and context are insufficient for a real assistant or RAG system. Later stages can keep the service contract while replacing the backend and evaluation method.
13. **What is your experimental claim?** Read the number from your own `artifacts/runs/.../report.json`. Claim only what this data, tokenizer, step count, and seed support. Compare with a baseline and change one variable at a time.
14. **How do data, compute, and parameters relate?** Capacity depends on parameters, learning signal on data, and operation count on compute. Increasing one side without balancing the others can waste resources. Treat scaling results as empirical, not universal laws.

## Portfolio exercise

Write a short report with:

- Problem and scope: character continuation on a synthetic bilingual corpus.
- Environment: OS, Python, PyTorch, CPU/GPU, threads, and seed.
- Data: source, license, hash, document/token counts per split, unknown rate, and shared-template limitation.
- Architecture: context, width, heads, layers, parameter count, and weight size.
- Baseline and two experiments: the one changed variable, best step, validation loss, and negative results.
- Final test: run after model selection and state the limited data independence.
- Deployment: endpoint, measured latency, error tests, and a publication URL only if actually public.
- Next steps: what must change for SFT, RAG, and a real product.

A suitable introduction is: “I implemented a small decoder-only GPT from scratch, trained it on documented data, evaluated it against a baseline and a controlled experiment, and served the checkpoint through a Docker-packaged API.”
