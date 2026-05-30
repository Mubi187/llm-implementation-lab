import torch
from torch.utils.data import DataLoader

from src.tokenizer import CharTokenizer
from src.dataset import ShakespeareDataset
from src.model import BigramLanguageModel
from src.utils import load_config


def main():
    # -------------------------
    # 1. Config
    # -------------------------
    config = load_config()
    device = config["training"]["device"]
    print(f"Device: {device}")

    # -------------------------
    # 2. Data
    # -------------------------
    with open(config["data"]["path"], "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Total characters: {len(text):,}")

    # -------------------------
    # 3. Tokenizer
    # -------------------------
    tokenizer = CharTokenizer(text)
    print(f"Tokenizer: {tokenizer}")
    print(f"Vocab: {''.join(tokenizer.chars)}")

    # -------------------------
    # 4. Encode & Split
    # -------------------------
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Total tokens: {len(data):,}")

    n = int(config["data"]["train_split"] * len(data))
    train_data = data[:n]
    val_data   = data[n:]
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    # -------------------------
    # 5. Datasets & Dataloaders
    # -------------------------
    block_size = config["model"]["block_size"]
    batch_size = config["training"]["batch_size"]

    train_dataset = ShakespeareDataset(train_data, block_size)
    val_dataset   = ShakespeareDataset(val_data,   block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # sanity check shapes
    xb, yb = next(iter(train_loader))
    print(f"\nBatch input shape:  {xb.shape}")    # (32, 128)
    print(f"Batch target shape: {yb.shape}")      # (32, 128)

    # -------------------------
    # 6. Model
    # -------------------------
    model = BigramLanguageModel(tokenizer.vocab_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: BigramLanguageModel")
    print(f"Parameters: {total_params:,}")

    # -------------------------
    # 7. Forward Pass Sanity Check
    # -------------------------
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)

    print(f"\n--- Sanity Check ---")
    print(f"Logits shape: {logits.shape}")
    print(f"Initial loss: {loss.item():.4f}  (expected ~4.17)")

    # -------------------------
    # 8. Generate Before Training
    # -------------------------
    print(f"\n--- Output BEFORE training ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=200)
    print(tokenizer.decode(generated[0].tolist()))

    print("\n" + "="*50)
    print("Setup complete. Trainer coming next.")
    print("="*50)


if __name__ == "__main__":
    main()
