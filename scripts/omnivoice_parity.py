"""Dumps Python OmniVoice values for the Rust port's parity test to match.

usage: omnivoice_parity.py <reference.wav> <out.safetensors>
Then: OMNIVOICE_ORACLE=<out.safetensors> cargo test \\
    --lib --no-default-features --features omnivoice,cuda,download matches_python -- --ignored --nocapture
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from safetensors.torch import save_file
from transformers import HiggsAudioV2TokenizerModel

from omnivoice import OmniVoice
from omnivoice.utils.audio import fade_and_pad_audio, load_audio, remove_silence
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig, _get_time_steps

REF_TEXT = (
    "The rabbit-hole went straight on like a tunnel for some way, and then dipped "
    "suddenly down, so suddenly that Alice had not a moment to think about stopping "
    "herself before she found herself falling down a very deep well."
)
TEXT = "Alice opened the door and found that it led into a small passage."

ESTIMATES = [
    "Hi.",
    "\u201cCurtsey as you\u2019re falling\u2014Do you see?\u201d",
    "It was 1865, and the Rabbit had 3 watches. " * 12,
]

ref, out = sys.argv[1:3]
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=getattr(torch, os.environ.get("ORACLE_DTYPE", "float16")))
tok = model.audio_tokenizer
sr = model.sampling_rate
print("audio tokenizer dtype", next(tok.parameters()).dtype, "sr", sr)

result = {}
wav = load_audio(ref, sr)
result["loaded"] = torch.from_numpy(wav.copy())
rms = float(np.sqrt(np.mean(wav**2)))
if 0 < rms < 0.1:
    wav = wav * 0.1 / rms
result["scaled"] = torch.from_numpy(wav.copy())
trimmed = remove_silence(wav, sr, mid_sil=200, lead_sil=100, trail_sil=200)
result["silence_removed"] = torch.from_numpy(trimmed.copy())
hop = tok.config.hop_length
clip = trimmed.shape[-1] % hop
cut = trimmed[:, :-clip] if clip > 0 else trimmed
x = torch.from_numpy(cut).to(tok.device).unsqueeze(0)

with torch.no_grad():
    resampled = torchaudio.functional.resample(x, sr, tok.config.semantic_sample_rate)
    result["resampled"] = resampled.cpu()
    semantic = tok._extract_semantic_features(x)
    result["semantic_features"] = semantic.cpu()
    e_semantic = tok.encoder_semantic(semantic.transpose(1, 2))
    result["e_semantic"] = e_semantic.cpu()
    lengths = tok._get_conv1d_output_lengths(x.shape[2], tok.acoustic_encoder)
    print("conv lengths", lengths, "semantic", e_semantic.shape[2])
    if lengths != e_semantic.shape[2]:
        e_acoustic = tok.acoustic_encoder(F.pad(x, (tok.pad, tok.pad)))
    else:
        e_acoustic = tok.acoustic_encoder(x)
    result["e_acoustic"] = e_acoustic.cpu()
    codes = tok.encode(x).audio_codes
    result["codes"] = codes.cpu().to(torch.int32)

# PyTorch's 32-bit GPU run of this encoder is the odd one out: its CPU and 64-bit
# GPU runs agree with each other, so the port is held to those.
accurate = HiggsAudioV2TokenizerModel.from_pretrained(tok.name_or_path, device_map="cpu", dtype=torch.float64)
with torch.no_grad():
    result["codes_accurate"] = accurate.encode(x.cpu().double()).audio_codes.cpu().to(torch.int32)
    result["semantic_accurate"] = accurate._extract_semantic_features(x.cpu().double()).float().cpu()
    result["acoustic_accurate"] = accurate.acoustic_encoder(x.cpu().double()).float().cpu()
del accurate

prompt = model.create_voice_clone_prompt(ref, ref_text=REF_TEXT)
assert torch.equal(prompt.ref_audio_tokens.cpu(), codes[0].cpu())
print("prompt", prompt.ref_audio_tokens.shape, "rms", prompt.ref_rms, repr(prompt.ref_text[-20:]))

est = model._estimate_target_tokens(TEXT, prompt.ref_text, prompt.ref_audio_tokens.size(-1))
slow = model._estimate_target_tokens(TEXT, prompt.ref_text, prompt.ref_audio_tokens.size(-1), speed=0.9)
print("estimate", est, "at 0.9", slow)

inputs = model._prepare_inference_inputs(
    TEXT, est, prompt.ref_text, prompt.ref_audio_tokens, None, None, True
)
result["input_ids"] = inputs["input_ids"][0].cpu().to(torch.int32)
result["audio_mask"] = inputs["audio_mask"][0].cpu().to(torch.int32)

# No random position choice, so both implementations should pick the same tokens.
captured = []
original = model._predict_tokens_with_scoring


def capture(c_logits, u_logits, gen_config):
    if not captured:
        captured.append((c_logits.cpu().clone(), u_logits.cpu().clone()))
    return original(c_logits, u_logits, gen_config)


model._predict_tokens_with_scoring = capture
config = OmniVoiceGenerationConfig(position_temperature=0.0, postprocess_output=False)
from omnivoice.models.omnivoice import GenerationTask

task = GenerationTask(
    batch_size=1,
    texts=[TEXT],
    target_lens=[est],
    langs=[None],
    instructs=[None],
    ref_texts=[prompt.ref_text],
    ref_audio_tokens=[prompt.ref_audio_tokens],
    ref_rms=[prompt.ref_rms],
)
with torch.no_grad():
    tokens = model._generate_iterative(task, config)[0]
result["det_tokens"] = tokens.cpu().to(torch.int32)
result["cond_logits_0"] = captured[0][0].float()[0, :, :4, :]
result["uncond_logits_0"] = captured[0][1].float()[0, :, :4, :]

with torch.no_grad():
    decoded = tok.decode(tokens.to(tok.device).unsqueeze(0)).audio_values[0].cpu().numpy()
result["decoded"] = torch.from_numpy(decoded.copy())
# A fixed signal, so post-processing can be compared without the model.
post = model._post_process_audio(
    decoded.copy(), ref_rms=prompt.ref_rms, gen_config=OmniVoiceGenerationConfig()
)
result["post_processed"] = torch.from_numpy(post.copy())

estimator = model.duration_estimator
result["char_weights"] = torch.tensor(
    [estimator._get_char_weight(chr(c)) if not 0xD800 <= c <= 0xDFFF else -1.0 for c in range(0x10000)],
    dtype=torch.float32,
)
timesteps = _get_time_steps(t_start=0.0, t_end=1.0, num_step=32, t_shift=0.1).tolist()
schedules = []
for t_len in (82, 300, 749):
    total = t_len * 8
    rem = total
    for step in range(32):
        num = rem if step == 31 else min(math.ceil(total * (timesteps[step + 1] - timesteps[step])), rem)
        schedules.append(int(num))
        rem -= int(num)
result["schedules"] = torch.tensor(schedules, dtype=torch.int32)
result["chunked"] = torch.tensor(
    [model._estimate_target_tokens(t, prompt.ref_text, 281) for t in ESTIMATES], dtype=torch.int32
)

metadata = {
    "ref_text": prompt.ref_text,
    "text": TEXT,
    "ref_rms": str(prompt.ref_rms),
    "estimate": str(est),
    "estimate_slow": str(slow),
}
save_file({k: v.contiguous() for k, v in result.items()}, out, metadata=metadata)
print(json.dumps({k: list(v.shape) for k, v in result.items()}))
