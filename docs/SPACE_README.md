---
title: Tiny LLM Course
emoji: 📘
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Tiny language model laboratory

A small character-level GPT with a simple UI and an API at `/docs`. It generates text continuations and is not a general-purpose assistant.

Suggested prompts: `The model ` or `Model `.

API: `POST /generate` with `prompt`, `max_new_tokens`, `temperature`, `top_k`, and `seed`. The `/health` endpoint reports service health and the checkpoint step.

The sample corpus was authored/generated for this project and split at document level. Sentence templates are shared across splits, so its quality is not general language quality. The source and educational architecture are released under MIT. No API key is required.
