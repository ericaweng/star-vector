# text2svg.py — copy-pasteable
import torch
from transformers import AutoModelForCausalLM

MODEL = "starvector/starvector-8b-im2svg"          # ✅ carries the text2svg finetune
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- load outer StarVector wrapper (it pulls vision + language modules) ----
sv = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).eval()

# ---- grab the inner SVG language model & tokenizer ------------------------
lm        = sv.model.svg_transformer             # StarCoder-like causal LM
tokenizer = sv.model.svg_transformer.tokenizer   #💡 shown in README :contentReference[oaicite:0]{index=0}

prompt = "Create an SVG of a simple rocket in flat style."

inputs = tokenizer(prompt, return_tensors="pt").to(device)
lm = lm.to(device).eval()

with torch.no_grad():
    ids = lm.generate(
        **inputs,
        max_new_tokens=1024,      # SVGs can be long — raise if your designs are complex
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id,
    )

svg_code = tokenizer.decode(ids[0], skip_special_tokens=True)

out_file = "rocket.svg"
with open(out_file, "w") as f:
    f.write(svg_code)

print(f"✓ SVG saved to {out_file}\nPreview:\n{svg_code[:200]} …")
