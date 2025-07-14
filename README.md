# Conditional GAN for Blond Hair Attribute Generation

This repository contains the code for building a Conditional Generative Adversarial Network (CGAN) that generates images of faces conditioned on the "blond hair" attribute. The goal of this project is to train a GAN where the output can be controlled based on a specific attribute, in this case, the presence of blond hair in facial images.

## Project Overview

In this project, I developed a Conditional GAN (CGAN) that takes both random noise and a label (blond or non-blond) as input. The model uses this conditional information to generate realistic face images with either blond or non-blond hair. This GAN consists of:
- A **generator** that generates synthetic images.
- A **critic** (or discriminator) that distinguishes between real and fake images.

### Key Features:
- **Conditional GAN**: The generator conditions on the label (blond or non-blond hair).
- **Customizable loss function**: Binary cross-entropy loss is used to train both the generator and discriminator.
- **Epoch-wise visualizations**: Generated images are saved at different training epochs to monitor the model's progress.

## Installation

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/your-username/conditional-gan-blond-hair-attribute.git
    cd conditional-gan-blond-hair-attribute
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. (Optional) Install TensorFlow if not already installed:
    ```bash
    pip install tensorflow
    ```

## Usage

1. **Training**: To begin training the model, simply run the `gan.py` script:
    ```bash
    python gan.py
    ```
   This will start the training loop, and the model will save images generated during each epoch. 

2. **Generated Images**: After every 10 epochs, generated images will be saved in the repository as PNG files. These images will help you observe how the model learns to generate faces conditioned on the "blond hair" attribute.

## Project Structure

- `gan.py`: The main script that contains the implementation of the Conditional GAN, including the generator, critic, and training loop.
- `generated_images_epoch_X.png`: Image files showing the results of the generator at each epoch (where X is the epoch number).
- `requirements.txt`: List of Python packages required to run the project.

## Results

At the end of the training process, the generator should be able to generate faces with either blond or non-blond hair depending on the input label. The model's performance can be evaluated by inspecting the generated images at different epochs.

## Contributing

If you would like to contribute to this project, feel free to open an issue or submit a pull request. Contributions are welcome!

