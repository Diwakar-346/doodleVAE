# interpolate.py
import matplotlib
# non-GUI backend to save plots
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import torch
import time
from tqdm import tqdm
import argparse
import os
import cv2

from src.config import *
from src.VAE import VAE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Train Device  \t= {device.upper()}")

def interpolate(model, latent_dim,):
    kernel_sharpen = np.array([ [0, -1,  0],
                                [-1, 5, -1],
                                [0, -1,  0]])

    print(f"Generating {N_KEYFRAMES} keyframes within range +/-{CONSTRAINED_Z_RANGE:.1f}...")
    keyframes_z = (torch.rand(N_KEYFRAMES, LATENT_DIM, device=device) * 2 - 1) * CONSTRAINED_Z_RANGE

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axis('off')

    with torch.no_grad():
        z_start_vis = keyframes_z[0:1]
        initial_img_tensor = model.decode(z_start_vis).cpu().squeeze()
        if initial_img_tensor.dim() == 3:
            initial_img_tensor = initial_img_tensor[0]
        initial_img = initial_img_tensor.numpy()

    im = ax.imshow(initial_img, cmap='gray_r', interpolation=INTERPOLATION_METHOD)
    title = ax.set_title("Frame 0")

    total_frames = (N_KEYFRAMES - 1) * N_INTER_STEPS
    print(f"Total frames to generate: {total_frames}")

    def update(frame):
        segment_idx = frame // N_INTER_STEPS
        step_in_segment = frame % N_INTER_STEPS
        z_start = keyframes_z[segment_idx]
        z_end = keyframes_z[min(segment_idx + 1, N_KEYFRAMES - 1)]
        alpha = step_in_segment / N_INTER_STEPS
        z_current = torch.lerp(z_start, z_end, alpha).unsqueeze(0)

        with torch.no_grad():
            try:
                generated_tensor = model.decode(z_current).cpu().squeeze()
                if generated_tensor.dim() == 3:
                    generated_tensor = generated_tensor[0]
                img_np = generated_tensor.numpy() if generated_tensor.dim() == 2 else np.zeros_like(initial_img)
            except Exception as e:
                img_np = np.zeros_like(initial_img)
        
        img_np = (img_np * 255).astype(np.uint8)  # Convert to 0–255
        img_np = cv2.filter2D(img_np, -1, kernel_sharpen)  # Apply sharpening
        img_np = img_np.astype(np.float32) / 255.0  # Back to float32 for plotting
        img_np = cv2.resize(img_np, (128, 128), interpolation=cv2.INTER_NEAREST)

        im.set_data(img_np)
        title.set_text(f"Frame {frame}/{total_frames}")
        return im, title
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    SAVE_FILENAME = os.path.join(MODEL_SAVE_DIR, f'latent_traversal_ld_{latent_dim}_{time.strftime("%Y%m%d_%H%M%S")}.mp4')

    writer = FFMpegWriter(fps=FPS, metadata=dict(artist='VAE Interpolator'))
    with writer.saving(fig, SAVE_FILENAME, dpi=100):
        for frame in tqdm(range(total_frames), desc="Generating frames"):
            update(frame)
            writer.grab_frame()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BetaVAE: Interpolate through the Latent Space")
    parser.add_argument('--model', '-i', type=str, required=True, help="Path to model .pth file")
    args = parser.parse_args()

    # === Load checkpoint and config ===
    print(f"Loading model checkpoint: {args.model}")
    checkpoint = torch.load(args.model, map_location=device)

    latent_dim = checkpoint.get('latent_dim', LATENT_DIM)

    print(f"→ Latent dim: {latent_dim}")

    # === Rebuild and load model ===
    model = VAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    interpolate(model, latent_dim=latent_dim)