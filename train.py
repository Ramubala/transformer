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


def parse_config_section(section):
    if isinstance(section, list):
        return {item["name"]: item["value"] for item in section}
    return section


def generate_text(model, tokenizer, seed_text, max_length=100, sequence_length=32):
    input_tokens = tokenizer.encode(seed_text)

    if len(input_tokens) > sequence_length:
        input_tokens = input_tokens[-sequence_length:]

    input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)

    generated_text = seed_text
    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_tensor)
            next_token = torch.argmax(output[:, -1, :], dim=-1).item()
            generated_text += tokenizer.decode([next_token])

            input_tokens.append(next_token)
            input_tokens = input_tokens[-sequence_length:]
            input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(
                input_tensor.device
            )

    return generated_text


if __name__ == "__main__":

    """Read corpus"""
    with open("data/corpus.txt", "r") as f:
        corpus_text = f.read()

    text = corpus_text[:100000]  # Use only the first 100k characters for training
    """Initialize tokenizer and dataset"""
    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    tokens = tokenizer.encode(text)

    """read config"""
    with open("config.yaml", "r") as f:
        config_master = yaml.safe_load(f)

    model_config = parse_config_section(config_master["model"])
    batch_size = model_config.get("batch_size")
    sequence_length = model_config.get("sequence_length")
    d_model = model_config.get("d_model")
    n_layers = model_config.get("n_layers")
    n_heads = model_config.get("n_heads")

    dataset = CustomDataset(tokens, block_size=sequence_length)

    """Create DataLoader"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = train_config.get("epochs", train_config.get("num_epochs"))

    """Training loop (simplified)"""
    for epoch in range(epochs):  # number of epochs
        model.train()
        for input_seq, target_seq in dataloader:
            # Forward pass
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            output = model(input_seq)
            loss = torch.nn.CrossEntropyLoss()(
                output.view(-1, tokenizer.vocab_size), target_seq.view(-1)
            )
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            eval_text = corpus_text[100000 : 100000 + sequence_length]
            generated_text = generate_text(
                model,
                tokenizer,
                eval_text,
                max_length=200,
                sequence_length=sequence_length,
            )
            logger.info(f"Seed text: {eval_text}")
            logger.info(f"prediction: {generated_text}")
