from flazoo.models.attentions import SlidingTileAttention2D, FullAttention
import torch
import time
# import amp
from torch.amp import autocast
seqlen = 16384
B, L, D = 2, seqlen, 224

x = torch.randn(B, L, D, requires_grad=True, device="cuda", dtype=torch.float32)

target = torch.randn(B, L, D, requires_grad=True, device="cuda", dtype=torch.float32)

attn1 = SlidingTileAttention2D(
    hidden_size=224,
    tile_size_h=8,
    tile_size_w=8,
    window_size_h=16,
    window_size_w=16,
    num_heads=14,
    seq_len=seqlen,
)

attn2 = FullAttention(
    hidden_size=224,
    num_heads=14,
)

attn1 = attn1.to(x.device)
attn2 = attn2.to(x.device)

# warmup
for i in range(10):
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        o = attn1(x)[0]

for i in range(10):
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        o = attn2(x)[0]


torch.cuda.synchronize()

time1 = time.time()

torch.cuda.synchronize()

for i in range(300):

    with autocast(device_type="cuda", dtype=torch.bfloat16):

        o = attn1(x)[0]

        loss = torch.nn.functional.mse_loss(o, target)
        loss.backward()

torch.cuda.synchronize()

time2 = time.time()

print(f"Time taken for STA 300 iterations with seqlen={seqlen}: ", time2 - time1)


time1 = time.time()

torch.cuda.synchronize()
for i in range(300):

    with autocast(device_type="cuda", dtype=torch.bfloat16):

        o = attn2(x)[0]

        loss = torch.nn.functional.mse_loss(o, target)
        loss.backward()
torch.cuda.synchronize()
time2 = time.time()
print(f"Time taken for FullAttention 300 iterations with seqlen={seqlen}: ", time2 - time1)