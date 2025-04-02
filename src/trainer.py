# src/trainer.py
import torch
import torch.nn.functional as F
import tqdm

def loss_function(recon_x, x, mu, logvar, beta):
    # calculates VAE loss (BCE + Beta * KLD)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

def train_epoch(model, dataloader, optimizer, device, beta, epoch_num):
    model.train()
    train_loss = 0.0
    num_samples = 0

    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch_num}", leave=False)
    for batch_idx, data in enumerate(progress_bar):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_samples += data.size(0)

        current_avg_loss = train_loss / num_samples if num_samples > 0 else 0
        progress_bar.set_postfix({'loss': f'{current_avg_loss:.4f}'})

    avg_loss = train_loss / num_samples if num_samples > 0 else 0
    return avg_loss