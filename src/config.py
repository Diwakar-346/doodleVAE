# src/config.py

# === model ===
MODEL_SAVE_DIR = "models"
EPOCHS          = 100
BATCH_SIZE      = 256
LATENT_DIM      = 32
BETA            = 2.0   # Beta-VAE factor
LEARNING_RATE   = 1e-3
CHECKPOINT_FREQ = 20    # saving model every N epochs
PLOT_FREQ       = 10
TRAIN_SPLIT     = 0.9
CHECKPOINT_FREQ = 5

# === visualization ===
N_KEYFRAMES = 8
N_INTER_STEPS = 50
EXTREME_Z_LIMIT = 3.0
CONSTRAINED_Z_RANGE = 2.0
INTERVAL_MS = 45
FPS = int(1000 / INTERVAL_MS)
VIDEO_WRITER = 'ffmpeg'
INTERPOLATION_METHOD = 'bilinear'
