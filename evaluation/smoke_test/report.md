# L4 GPU Smoke Test Report

**Generated**: 2026-09-24T19:04:07.358151+00:00

## Hardware

| Metric | Value |
|--------|-------|
| GPU | NVIDIA L4 |
| Total VRAM | 22.03 GB |
| CUDA | 12.8 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 5.17.0 |
| PEFT | 0.21.0 |
| bitsandbytes | 0.50.2 |

## Test: synthetic

| Metric | Value |
|--------|-------|
| Examples | 5 |
| Max Length | 256 |
| Steps | 5 |
| Grad Accum | 1 |
| Trainable Params | 33,030,144 (1.48%) |
| VRAM After Model Load (alloc) | 2.494 GB |
| VRAM After Model Load (reserved) | 2.539 GB |
| VRAM After Model Load (peak alloc) | 2.53 GB |
| VRAM After Model Load (peak reserved) | 2.539 GB |
| VRAM After kbit Prep (alloc) | 3.219 GB |
| VRAM After kbit Prep (reserved) | 3.988 GB |
| VRAM After kbit Prep (peak alloc) | 3.943 GB |
| VRAM After kbit Prep (peak reserved) | 3.988 GB |
| VRAM After QLoRA Attach (alloc) | 3.342 GB |
| VRAM After QLoRA Attach (reserved) | 4.111 GB |
| VRAM After QLoRA Attach (peak alloc) | 3.943 GB |
| VRAM After QLoRA Attach (peak reserved) | 4.111 GB |
| VRAM After Training (Peak) (alloc) | 3.389 GB |
| VRAM After Training (Peak) (reserved) | 5.004 GB |
| VRAM After Training (Peak) (peak alloc) | 4.639 GB |
| VRAM After Training (Peak) (peak reserved) | 5.004 GB |
| Training Loss | 2.3691 |
| Loss Decreasing | YES |
| Training Step | PASS |
| Checkpoint Save | PASS |
| Checkpoint Reload | PASS |
| Inference After Reload | PASS |

### Inference Output Sample
```
A student asks about gravity.

### Output:
You must explain gravity in a simple, clear, and engaging way.

### Constraints:
- Keep it simple and avoid technical terms.
- Use everyday examples.
- Make 
```

### Loss History
- Step 1: 3.0329062938690186
- Step 2: 2.57600474357605
- Step 3: 2.5833218097686768
- Step 4: 1.9100924730300903
- Step 5: 1.7431621551513672

## Test: real

| Metric | Value |
|--------|-------|
| Examples | 20 |
| Max Length | 2048 |
| Steps | 10 |
| Grad Accum | 4 |
| Trainable Params | 33,030,144 (1.48%) |
| VRAM After Model Load (alloc) | 3.961 GB |
| VRAM After Model Load (reserved) | 4.004 GB |
| VRAM After Model Load (peak alloc) | 3.998 GB |
| VRAM After Model Load (peak reserved) | 4.004 GB |
| VRAM After kbit Prep (alloc) | 4.686 GB |
| VRAM After kbit Prep (reserved) | 5.453 GB |
| VRAM After kbit Prep (peak alloc) | 5.41 GB |
| VRAM After kbit Prep (peak reserved) | 5.453 GB |
| VRAM After QLoRA Attach (alloc) | 4.809 GB |
| VRAM After QLoRA Attach (reserved) | 5.576 GB |
| VRAM After QLoRA Attach (peak alloc) | 5.41 GB |
| VRAM After QLoRA Attach (peak reserved) | 5.576 GB |
| VRAM After Training (Peak) (alloc) | 4.841 GB |
| VRAM After Training (Peak) (reserved) | 11.025 GB |
| VRAM After Training (Peak) (peak alloc) | 9.898 GB |
| VRAM After Training (Peak) (peak reserved) | 11.025 GB |
| Training Loss | 1.6818 |
| Loss Decreasing | YES |
| Training Step | PASS |
| Checkpoint Save | PASS |
| Checkpoint Reload | PASS |
| Inference After Reload | PASS |

### Inference Output Sample
```
[Response content]

### Evaluation:
[Pass/Fail]

### Explanation:
[Why it passed or failed]

### Response:
[Response content]

### Evaluation:
[Pass/Fail]

### Explanation:
[Why it passed or failed]


```

### Loss History
- Step 1: 2.62748384475708
- Step 2: 2.225736618041992
- Step 3: 2.0231704711914062
- Step 4: 1.7828528881072998
- Step 5: 1.637380838394165
- Step 6: 1.4055781364440918
- Step 7: 1.3546783924102783
- Step 8: 1.2894433736801147
- Step 9: 1.250105619430542
- Step 10: 1.221604824066162
