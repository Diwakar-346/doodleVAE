# src/visualize.py
import os
import matplotlib
# use backend that doesn't require a GUI
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

def plot_loss_history(train_history, title="Training Loss History", save_path=None):
    if not train_history:
        print("Warning: No training history provided to plot.")
        return

    epochs = range(1, len(train_history) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_history, 'b-o', label='Training Loss')

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss per Sample")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved loss history plot to {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)