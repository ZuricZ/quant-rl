#!/usr/bin/env bash
# Serves a Qwen3 model with vLLM on this machine's GPU, as an OpenAI-compatible
# endpoint with tool calling, for the *-vllm entries of runs/models.toml.
#
#   scripts/serve_vllm.sh [MODEL] [PORT]     # default Qwen/Qwen3-4B on 8000
#
# Needs a CUDA GPU and vLLM (`uv tool install vllm`, or `pip install vllm` in
# its own environment: it is not a dependency of this project).
#
# Context: Qwen3's native window is 32,768 tokens. A 30-turn episode with
# thinking can outgrow it; episodes that do end early (see results.py's "How
# episodes ended"). If many do, extend it with YaRN as the Qwen3 model card
# describes and raise --max-model-len.
#
# Tool calls: Qwen3 emits Hermes-style <tool_call> tags, which the hermes
# parser turns into OpenAI tool calls. The reasoning parser moves <think>
# blocks out of the message content.
#
# If the GPU is on another machine, forward the port so localhost:8000
# reaches it (ssh -N -L 8000:localhost:8000 gpu-host), or edit base_url.
set -euo pipefail

MODEL="${1:-Qwen/Qwen3-4B}"
PORT="${2:-8000}"

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len 32768 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3
