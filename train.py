import pandas as pd
from src.Tokenizer import CharTokenizer
from src.Dataset import CustomDataset
from torch.utils.data import DataLoader
from src.Model import Model
import yaml
import torch
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.info(torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# GPU optimizations
if device == "cuda":
    torch.backends.cudnn.benchmark = True


def parse_config_section(section):
    if isinstance(section, list):
        return {item["name"]: item["value"] for item in section}
    return section


def generate_text(model, tokenizer, seed_text, max_length=100, sequence_length=32):
    input_tokens = tokenizer.encode(seed_text)

    if len(input_tokens) > sequence_length:
        input_tokens = input_tokens[-sequence_length:]

    # Create tensor once on GPU, reuse it
    input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=device)

    generated_text = seed_text
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_tensor)
            next_token = torch.argmax(output[:, -1, :], dim=-1).item()
            generated_text += tokenizer.decode([next_token])

            input_tokens.append(next_token)
            input_tokens = input_tokens[-sequence_length:]
            # Update tensor in-place on GPU to avoid CPU→GPU transfer
            input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=device)

    return generated_text


if __name__ == "__main__":

    """Read corpus"""
    with open("data/corpus.txt", "r") as f:
        corpus_text = f.read()

    n = len(corpus_text)
    logger.info(f"Corpus length: {n/1000000:.2f} million characters")

    text = corpus_text[: int(0.8 * n)]  # train_text
    val_text = corpus_text[int(0.8 * n) : int(0.9 * n)]  # validation_text

    """Initialize tokenizer and dataset"""
    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    tokens = tokenizer.encode(text)
    val_tokens = tokenizer.encode(val_text)

    """read config"""
    with open("config.yaml", "r") as f:
        config_master = yaml.safe_load(f)

    model_config = parse_config_section(config_master["model"])
    batch_size = model_config.get("batch_size")
    sequence_length = model_config.get("sequence_length")
    d_model = model_config.get("d_model")
    n_layers = model_config.get("n_layers")
    n_heads = model_config.get("n_heads")

    test_text = corpus_text[int(0.92 * n) : int(0.92 * n + sequence_length)]

    dataset = CustomDataset(tokens, block_size=sequence_length)
    val_dataset = CustomDataset(val_tokens, block_size=sequence_length)

    """Create DataLoader"""
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device == "cuda"),
        num_workers=4,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device == "cuda"),
        num_workers=4,
        persistent_workers=True,
    )

    """Initialize model"""
    model = Model(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        block_size=sequence_length,
    )

    model.to(device)
    model.train()

    train_config = parse_config_section(config_master["training"])
    learning_rate = train_config.get("learning_rate")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0.001
    )
    epochs = train_config.get("epochs", train_config.get("num_epochs"))

    best_val_loss = float("inf")
    loss_tracker = []
    scaler = torch.amp.GradScaler(device)
    criterion = torch.nn.CrossEntropyLoss()  # Create once, reuse

    """Training loop (simplified)"""
    for epoch in range(epochs):  # number of epochs
        model.train()
        for input_seq, target_seq in dataloader:
            # Forward pass
            input_seq = input_seq.to(device, non_blocking=True)
            target_seq = target_seq.to(device, non_blocking=True)

            if device == "cuda" and scaler is not None:
                with torch.amp.autocast(device):
                    output = model(input_seq)
                    loss = criterion(
                        output.view(-1, tokenizer.vocab_size), target_seq.view(-1)
                    )
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(input_seq)
                loss = criterion(
                    output.view(-1, tokenizer.vocab_size), target_seq.view(-1)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        optimizer.zero_grad()

        """Validation loop (simplified)"""
        model.eval()
        val_loss = 0.0
        eval_iters = train_config.get("eval_iters")
        with torch.no_grad():
            for i, (val_input_seq, val_target_seq) in enumerate(val_dataloader):
                if i >= eval_iters:
                    break
                val_input_seq = val_input_seq.to(device, non_blocking=True)
                val_target_seq = val_target_seq.to(device, non_blocking=True)
                val_output = model(val_input_seq)
                val_loss += criterion(
                    val_output.view(-1, tokenizer.vocab_size), val_target_seq.view(-1)
                ).item()
        val_loss /= eval_iters

        logger.info(
            f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}"
        )
        loss_tracker.append(
            {"epoch": epoch + 1, "train_loss": loss.item(), "val_loss": val_loss}
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            logger.info(f"New best model saved with val loss: {best_val_loss:.4f}")

        """Sample text every 10 epochs to save time"""
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                generated_text = generate_text(
                    model,
                    tokenizer,
                    test_text,
                    max_length=200,
                    sequence_length=sequence_length,
                )
                logger.info(f"Seed text: {test_text}")
                logger.info(f"prediction: {generated_text}")

    # Save training history
    history_df = pd.DataFrame(loss_tracker)
    history_df.to_csv("training_history.csv", index=False)
