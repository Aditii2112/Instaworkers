#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy_to_ollama.sh — Sovereign Sentinel deployment pipeline
#
# Steps:
#   1. Fuse LoRA adapters into the base PaliGemma 2 model (mlx_lm.fuse)
#   2. Convert the fused model to GGUF format        (llama.cpp)
#   3. Generate a Modelfile for Ollama
#   4. Create the Ollama model (ollama create)
#
# Prerequisites:
#   - mlx-lm installed            (pip install mlx-lm)
#   - llama.cpp cloned at ../llama.cpp   (with convert_hf_to_gguf.py)
#   - ollama installed & running  (ollama serve)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Use project venv Python if present (has mlx-vlm); otherwise system python3
if [ -x "${SCRIPT_DIR}/venv/bin/python3" ]; then
  PYTHON="${SCRIPT_DIR}/venv/bin/python3"
else
  PYTHON="python3"
fi

# ── Configurable paths ───────────────────────────────────────────────────────
BASE_MODEL="${SCRIPT_DIR}/models/paligemma2-base"
ADAPTER_DIR="${SCRIPT_DIR}/adapters"
FUSED_MODEL="${SCRIPT_DIR}/models/paligemma2-fused"
BASE_EXPORT_DIR="${SCRIPT_DIR}/models/paligemma2-base-export"
GGUF_OUTPUT="${SCRIPT_DIR}/models/sovereign-sentinel.gguf"
# Override with: LLAMA_CPP_DIR=/path/to/llama.cpp ./deploy_to_ollama.sh
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${SCRIPT_DIR}/../llama.cpp}"
MODEL_NAME="sovereign-sentinel"
MODELFILE="${SCRIPT_DIR}/Modelfile"
QUANTIZATION="q8_0"

# Use base model only (no fine-tuned LoRA). Keeps adapters/fused files; deploys HF base only.
USE_BASE_ONLY="${USE_BASE_ONLY:-0}"
[ "${1:-}" = "--base-only" ] && USE_BASE_ONLY=1

# ── Preflight checks ─────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Sovereign Sentinel — Ollama Deployment Pipeline"
echo "═══════════════════════════════════════════════════════════════════════"

fail() { echo "✗  $1" >&2; exit 1; }

[ -d "$BASE_MODEL" ]   || fail "Base model not found: $BASE_MODEL"
[ -d "$LLAMA_CPP_DIR" ] || fail "llama.cpp not found at: $LLAMA_CPP_DIR"
command -v ollama >/dev/null 2>&1 || fail "ollama CLI not found — brew install ollama"
"$PYTHON" -c "import mlx_vlm" 2>/dev/null || fail "mlx-vlm not installed — pip install mlx-vlm (or activate venv)"

if [ "$USE_BASE_ONLY" = "1" ]; then
  echo "  Mode        : BASE ONLY (Hugging Face model, no LoRA)"
else
  [ -d "$ADAPTER_DIR" ] || fail "Adapter dir not found: $ADAPTER_DIR"
  echo "  Mode        : Fine-tuned (fuse LoRA then deploy)"
fi
echo "  Base model  : $BASE_MODEL"
echo "  GGUF output : $GGUF_OUTPUT"
echo "  Quantization: $QUANTIZATION"
echo ""

# ── Step 1: Prepare model for GGUF (either export base or fuse LoRA) ─────────
if [ "$USE_BASE_ONLY" = "1" ]; then
  echo "▸ Step 1/4 — Exporting base model (no adapters) for GGUF …"
  "$PYTHON" "${SCRIPT_DIR}/export_base_to_hf.py" \
      --model "$BASE_MODEL" \
      --save-path "$BASE_EXPORT_DIR"
  echo "  ✓ Base export saved to $BASE_EXPORT_DIR"
  HF_FOR_GGUF="$BASE_EXPORT_DIR"
else
  echo "▸ Step 1/4 — Fusing LoRA adapters into base model …"
  "$PYTHON" "${SCRIPT_DIR}/fuse_adapters.py" \
      --model "$BASE_MODEL" \
      --adapter-path "$ADAPTER_DIR" \
      --save-path "$FUSED_MODEL" \
      --de-quantize
  echo "  ✓ Fused model saved to $FUSED_MODEL"
  HF_FOR_GGUF="$FUSED_MODEL"
fi
echo ""

# ── Ensure tokenizer.model for GGUF converter (Gemma2 needs SentencePiece) ───
if [ ! -f "${HF_FOR_GGUF}/tokenizer.model" ]; then
  echo "▸ tokenizer.model not in model dir; downloading from HuggingFace …"
  "$PYTHON" -c "
import sys
from pathlib import Path
out_dir = Path(sys.argv[1])
try:
    from huggingface_hub import hf_hub_download
    hf_hub_download('google/gemma-2-2b', 'tokenizer.model', local_dir=str(out_dir))
    print('  ✓ tokenizer.model downloaded to', out_dir)
except Exception as e:
    print('  ✗ Download failed:', e)
    print('  Put tokenizer.model in', out_dir, 'or in', sys.argv[2], 'and re-run.')
    sys.exit(1)
" "$HF_FOR_GGUF" "$BASE_MODEL" || exit 1
else
  echo "  ✓ tokenizer.model present"
fi
echo ""

# ── Step 2: Convert to GGUF ──────────────────────────────────────────────────
echo "▸ Step 2/4 — Converting model to GGUF ($QUANTIZATION) …"
CONVERT_SCRIPT="${LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
[ -f "$CONVERT_SCRIPT" ] || fail "convert_hf_to_gguf.py not found in $LLAMA_CPP_DIR"

"$PYTHON" "$CONVERT_SCRIPT" \
    "$HF_FOR_GGUF" \
    --outfile "$GGUF_OUTPUT" \
    --outtype "$QUANTIZATION"
echo "  ✓ GGUF model written to $GGUF_OUTPUT"
echo ""

# ── Step 3: Generate Modelfile ───────────────────────────────────────────────
echo "▸ Step 3/4 — Generating Ollama Modelfile …"
cat > "$MODELFILE" <<'MODELFILE_EOF'
FROM ./models/sovereign-sentinel.gguf

TEMPLATE """{{ if .System }}<start_of_turn>system
{{ .System }}<end_of_turn>
{{ end }}{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ end }}{{ .Response }}<end_of_turn>"""

PARAMETER temperature 0.4
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 1024
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<eos>"

SYSTEM """You are Sovereign Sentinel, an edge AI agent that performs visual inspection of warehouse and industrial feeds. For every frame you receive, produce a structured response with three sections:

SEE: Describe exactly what you observe — objects, defects, anomalies, environmental hazards.
REASON: Explain your analysis — why this matters, risk level, relevant safety codes.
ACT: Output a JSON array of tool calls to execute, e.g. [{"tool": "flag_violation", "params": {"type": "...", "severity": "..."}}].

Be precise, safety-first, and never fabricate observations."""
MODELFILE_EOF
echo "  ✓ Modelfile written to $MODELFILE"
echo ""

# ── Step 4: Create Ollama model ──────────────────────────────────────────────
echo "▸ Step 4/4 — Creating Ollama model '$MODEL_NAME' …"
ollama create "$MODEL_NAME" -f "$MODELFILE"
echo "  ✓ Ollama model '$MODEL_NAME' created"
echo ""

# ── Done ─────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════════════"
echo "  Deployment complete!"
echo ""
echo "  Test with:"
echo "    ollama run $MODEL_NAME 'Analyze this warehouse feed.'"
echo ""
echo "  API endpoint:"
echo "    curl http://localhost:11434/api/generate -d '{"
echo "      \"model\": \"$MODEL_NAME\","
echo "      \"prompt\": \"Analyze this warehouse feed.\"'"
echo "    }'"
echo "═══════════════════════════════════════════════════════════════════════"
