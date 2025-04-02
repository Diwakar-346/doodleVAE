<a id="readme-top"></a>

<br />
<div align="center">
  <!-- <a href="https://github.com/yassa9/doodleVAE">
    <img src="images/logo.jpg" alt="Logo" width="80" height="80">
  </a> -->

  <h1 align="center" style="font-size: 60px;">doodleVAE</h1>

  <p align="center">
    Yet another disentangled VAE ... but for quick drawing doodles.
    <br />

[![GIF shot][product-screenshot]](https://github.com/yassa9/doodleVAE)

  </p>
</div>


<!-- ABOUT THE PROJECT -->
## The Model

This project implements a `Variational Autoencoder (VAE)` trained to generate hand-drawn doodles. It learns a compressed latent representation of doodles from the `Quick, Draw! dataset` and uses it to generate new, human-like sketches.

### The model consists of:

- [x] Convolutional encoder that compresses input images into a latent vector
- [x] Reparameterization layer to sample from the latent space
- [x] Convolutional decoder that reconstructs images from latent vectors
- [x] Latent space exploration, saved as an animation
- [x] Loss plotting

The training pipeline supports configurable hyperparameters (e.g. latent dimension, beta, batch size, epochs) through a configuration file or command-line arguments.

<p align="right">(<a href="#readme-top">Back Top</a>)</p>

### Built With

* [![Python][python]][python-url]
* [![PyTorch][pytorch]][pytorch-url]
* [![Numpy][numpy]][numpy-url]
* [![Matplotlib][matplotlib]][matplotlib-url]
* [![openCV][opencv]][opencv-url]

<p align="right">(<a href="#readme-top">Back Top</a>)</p>

## Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

Ensure you have Python installed (>= 3.8 recommended).  
You can install it from [python.org](https://www.python.org/).

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yassa9/doodleVAE.git
   cd doodleVAE
   ```

2. **Install dependencies**

   You can install everything with pip:
   ```bash
   pip install torch torchvision matplotlib numpy
   ```

3. **Prepare your dataset**

   - Provide a file `<file>.npy` path using `--data-path`.
   - You can get data from [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset).

### Training

To train the model, run:

```bash
python train.py --data-path path/to/<file>.npy
```

You can customize training with command-line arguments:

```bash
python train.py --data-path cat.npy --epochs 50 --latent-dim 20 --beta 4
```

| Argument        | Description                             |
|----------------|-----------------------------------------|
| `--epochs`      | Number of training epochs               |
| `--batch-size`  | Training batch size                     |
| `--latent-dim`  | Dimensionality of latent space          |
| `--beta`        | Beta value for KL divergence term       |
| `--lr`          | Learning rate                           |
| `--save-dir`    | Directory to save model and plots       |
| `--no-explore`  | Skip final latent interpolation animation |


<p align="right">(<a href="#readme-top">Back Top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[product-screenshot]: images/gifshot.gif

[python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
[pytorch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[pytorch-url]: https://pytorch.org/
[numpy]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[numpy-url]: https://numpy.org/
[matplotlib]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[matplotlib-url]: https://matplotlib.org/
[opencv]: https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white
[opencv-url]: https://opencv.org/