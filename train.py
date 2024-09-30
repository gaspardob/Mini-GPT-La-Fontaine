import torch
import torch.optim as optim
from transfomer-decoder import LanguageModel
from utils import get_batch, estimate_loss
import torch
import torch.optim as optim
from model import LanguageModel
from utils import get_batch, estimate_loss

batch_size = 64
block_size = 128
max_iters = 100
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
weight_decay = 1e-2
grad_clip = 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('poems.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda l: ''.join([itos.get(i, '') for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

model = LanguageModel(vocab_size).to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M paramètres")

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, gamma=0.98)

for iter in range(1, max_iters + 1):
    xb, yb = get_batch(train_data, batch_size, block_size, device)
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()

    if iter % eval_interval == 0 or iter == max_iters:
        train_loss = estimate_loss(model, train_data, eval_iters, batch_size, block_size, device)
        val_loss = estimate_loss(model, val_data, eval_iters, batch_size, block_size, device)
        print(f"Step {iter}: training loss {train_loss:.4f}, validation loss{val_loss:.4f}, lr {scheduler.get_last_lr()[0]:.6f}")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=2000, temperature=0.8)
print(decode(generated[0].tolist()))
