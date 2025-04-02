# train.py
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
import time
import tqdm
import os

from src.config import *
from src.data_loader import get_data_loaders
from src.VAE import VAE
from src.trainer import train_epoch
from src.visualize import plot_loss_history
from interpolate import interpolate

def parsing():
    parser = argparse.ArgumentParser(description="BetaVAE: Train a disentangled VAE on Google Doodle Data")
    parser.add_argument('--data-path',   '-i' , type=str,   required=True,           help=f"Path to .npy data file)")
    parser.add_argument('--save-dir',    '-o' , type=str,   default=MODEL_SAVE_DIR,  help=f"Directory to save models (default: {MODEL_SAVE_DIR})")
    parser.add_argument('--epochs',      '-e' , type=int,   default=EPOCHS,          help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument('--batch-size',  '-bs', type=int,   default=BATCH_SIZE,      help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument('--latent-dim',  '-ld', type=int,   default=LATENT_DIM,      help=f"Latent dimension size (default: {LATENT_DIM})")
    parser.add_argument('--beta',        '-B' , type=float, default=BETA,            help=f"Beta value for KLD term (default: {BETA})")
    parser.add_argument('--lr',                 type=float, default=LEARNING_RATE,   help=f"Learning rate (default: {LEARNING_RATE})")
    parser.add_argument('--no-explore',                     action='store_true',     help="Disable latent space exploration step")
    return parser.parse_args()

def main(args):
    data_path = args.data_path
    save_dir = args.save_dir if args.save_dir else MODEL_SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)
    epochs = args.epochs if args.epochs is not None else EPOCHS
    batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZE
    latent_dim = args.latent_dim if args.latent_dim is not None else LATENT_DIM
    beta = args.beta if args.beta is not None else BETA
    lr = args.lr if args.lr is not None else LEARNING_RATE
    # ckpt_freq = args.checkpoint_freq if args.checkpoint_freq is not None else CHECKPOINT_FREQ

    print("\n=== Model Configs ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Train Device  \t= {device.upper()}")
    print(f"Epochs        \t= {epochs}")
    print(f"Batch Size    \t= {batch_size}")
    print(f"Latent Dim    \t= {latent_dim}") 
    print(f"KLD Beta      \t= {beta}")
    print(f"Learning Rate \t= {lr}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"vae_ld{latent_dim}_beta{beta}_{timestamp}"

    train_loader, _ = get_data_loaders(
        dataset=data_path,
        train_split_ratio=TRAIN_SPLIT,
        batch_size=batch_size,
    )

    print("\n=== Initializing Model ===")
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,} \n")

    # === Training Loop ===
    train_losses = []
    start_train_time = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        avg_train_loss = train_epoch(model, train_loader, optimizer, device, beta, epoch)
        train_losses.append(avg_train_loss)

        epoch_time = time.time() - epoch_start_time
        print(f"====> Epoch: {epoch}/{epochs} | Avg. train loss: {avg_train_loss:.4f} | Time: {epoch_time:.2f}s")

    total_train_time = time.time() - start_train_time
    print(f"\n=== Training Complete ===")
    print(f"Total Training Time: {total_train_time:.2f}s ({total_train_time/60:.2f} minutes)")

    # === Save Final Model ===
    final_model_path = os.path.join(save_dir, f"{run_name}_final.pth")
    print(f"\n=== Saving Final Model State Dictionary ===")
    print(f"Path: {final_model_path}")
    torch.save(model.state_dict(), final_model_path)
    checkpoint = {
        'state_dict': model.state_dict(),
        'latent_dim': latent_dim,
    }
    torch.save(checkpoint, final_model_path)

    # === Plot Final Loss History ===
    if train_losses:
        print("\n=== Generating Final Loss Plot ===")
        loss_plot_save_path = os.path.join(save_dir, f"{run_name}_loss.png") if save_dir else None
        if loss_plot_save_path:
            plot_loss_history(
                train_history=train_losses,
                title=f"Loss ({run_name})",
                save_path=loss_plot_save_path
            )

    # === Interpolation in Latent Space ===
    if not args.no_explore:
        interpolate(model, latent_dim=latent_dim)

if __name__ == "__main__":
    args = parsing()
    main(args)
