if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    # Load training history from CSV
    history = pd.read_csv("training_history.csv")

    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs Train and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
