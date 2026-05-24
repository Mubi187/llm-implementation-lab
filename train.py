import torch
from torch.utils.data import DataLoader
from src.tokenizer import CharTokenizer
from src.dataset import ShakespeareDataset
from src.utils import load_config

config = load_config()

# load text
with open(config["data"]["path"], "r", encoding="utf-8") as f:
    text = f.read()

# tokenize
tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
print(f"Total tokens: {len(data)}")        # ~1M

# train / val split
n = int(config["data"]["train_split"] * len(data))
train_data = data[:n]
val_data   = data[n:]
print(f"Train: {len(train_data)} | Val: {len(val_data)}")

# datasets
block_size = config["model"]["block_size"]
train_dataset = ShakespeareDataset(train_data, block_size)
val_dataset   = ShakespeareDataset(val_data,   block_size)

# dataloaders
batch_size = config["training"]["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

# sanity check
xb, yb = next(iter(train_loader))
print(f"Input shape:  {xb.shape}")   # (32, 128)
print(f"Target shape: {yb.shape}")   # (32, 128)

# visualize one sample
print("\n--- One training sample ---")
print(f"Input:  {xb[0].tolist()}")
print(f"Target: {yb[0].tolist()}")
print(f"Decoded input: {tokenizer.decode(xb[0].tolist())}")
