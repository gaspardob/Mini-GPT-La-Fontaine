import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparamètres
batch_size = 64  # combien de séquences indépendantes allons-nous traiter en parallèle ?
block_size = 128  # quelle est la longueur de contexte maximale pour les prédictions ?
max_iters = 3000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

eval_iters = 200
n_embd = 200
n_head = 5
n_layer = 4
dropout = 0.2


torch.manual_seed(1337)

# Chargement du texte
with open('laf.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encodeur : prend une chaîne, renvoie une liste d'entiers
decode = lambda l: ''.join([itos[i] for i in l]) # décodeur : prend une liste d'entiers, renvoie une chaîne

# Division des données en ensemble d'entraînement et de validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # 90% pour l'entraînement, le reste pour la validation
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # génère un petit batch de données avec des entrées x et des cibles y
    data = train_data si split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ une tête de self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # Calcul des scores d'attention ("affinités") en utilisant torch.einsum
        wei = torch.einsum('btc,bsc->bts', q, k) * C**-0.5 # (B, T, T)  # NORMALISATIOn par sqrt(C)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
         # ou sinon:
        # mask = torch.arange(T)[None, :] > torch.arange(T)[:, None]
        # wei = wei.masked_fill(mask, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # Agrégation pondérée des valeurs
        v = self.value(x) # (B,T,C)
        out = torch.einsum('bts,bsc->btc', wei, v) # (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ plusieurs têtes de self-attention en parallèle """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ une simple couche linéaire suivie d'une non-linéarité """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Bloc Transformer : communication suivie de calculs """

    def __init__(self, n_embd, n_head):
        # n_embd : dimension des embeddings, n_head : nombre de têtes
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    """ Modèle de langage basé sur les bigrammes """

    def __init__(self):
        super().__init__()
        # chaque token lit directement les logits pour le token suivant à partir d'une table de lookup
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # normalisation finale
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx et targets sont tous deux des tenseurs (B,T) d'entiers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        # idx est un tableau (B, T) des indices dans le contexte actuel
        for _ in range(max_new_tokens):
            # couper idx aux derniers block_size tokens
            idx_cond = idx[:, -block_size:]
            # obtenir les prédictions
            logits, loss = self(idx_cond) #(B,taille_phrase) vers (B,taille_phrase, vocab_size)
            # ne se concentrer que sur la dernière étape temporelle
            logits = logits[:, -1, :] / temperature # devient (B, C)
            #PLUTOT QUE [:][-1][:]:  cela sélectionne le dernier élément du batch entier, et non pas le dernier token pour chaque élément du batch.
            # appliquer softmax pour obtenir les probabilités
            probs = F.softmax(logits, dim=-1) # (B, C)
            # échantillonner à partir de la distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # ajouter l'indice échantillonné à la séquence en cours
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


if __name__ == "__main__":
  model = BigramLanguageModel()
  m = model.to(device)

  print(sum(p.numel() for p in m.parameters())/1e6, 'M paramètres')
  
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  
  for iter in range(max_iters):
      si iter % eval_interval == 0 ou iter == max_iters - 1:
          losses = estimate_loss()
          print(f"étape {iter}: perte d'entraînement {losses['train']:.4f}, perte de validation {losses['val']:.4f}")
  
      xb, yb = get_batch('train')

      logits, loss = model(xb, yb)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
  
  context = torch.zeros((1, 1), dtype=torch.long, device=device)
  print(decode(m.generate(context, max_new_tokens=2000, temperature=0.8)[0].tolist()))
