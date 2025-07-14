import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Define the generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(102,)))  # Changed to (102,)
    model.add(layers.BatchNormalization())  
    model.add(layers.Dense(256, activation='relu'))  
    model.add(layers.BatchNormalization())  
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.BatchNormalization())  
    model.add(layers.Dense(784, activation='sigmoid'))  
    model.add(layers.Reshape((28, 28, 1)))  # Output a 28x28 grayscale image.
    return model

# Define the critic model
def build_critic():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))  
    model.add(layers.Flatten())  
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1))  # Output a single value (real or fake).
    return model

# Define the Conditional GAN class
class ConditionalGAN(tf.keras.Model):
    def __init__(self, generator, critic):
        super(ConditionalGAN, self).__init__()
        self.generator = generator
        self.critic = critic

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images, labels):
        noise = tf.random.normal([real_images.shape[0], 100])  
        labels = tf.one_hot(labels, depth=2)  
        conditioned_noise = tf.concat([noise, labels], axis=1)  

        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            fake_images = self.generator(conditioned_noise, training=True)

            real_output = self.critic(real_images, training=True)  
            fake_output = self.critic(fake_images, training=True)  

            d_loss = self.loss_fn(tf.ones_like(real_output), real_output) + \
                     self.loss_fn(tf.zeros_like(fake_output), fake_output)
            g_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)

        d_gradients = d_tape.gradient(d_loss, self.critic.trainable_variables)
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)

        self.d_optimizer.apply_gradients(zip(d_gradients, self.critic.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# Define the function to visualize the generated images
def plot_generated_images(epoch, generator, examples=10, noise_dim=100):
    noise = np.random.normal(0, 1, (examples, noise_dim))  # Generate random noise.
    labels = np.random.randint(0, 2, examples)  # Random labels for conditioning.
    labels = tf.one_hot(labels, depth=2)  # One-hot encode the labels.
    conditioned_noise = np.concatenate([noise, labels], axis=1)  # Concatenate noise and labels.

    generated_images = generator.predict(conditioned_noise)  # Generate images from the conditioned noise.

    plt.figure(figsize=(10, 10))
    for i in range(examples):
        plt.subplot(1, examples, i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')  # Display in grayscale.
        plt.axis('off')  # Remove axis.
    plt.tight_layout()
    plt.savefig(f"generated_images_epoch_{epoch}.png")  # Save the generated images.
    plt.show()

# Initialize the models
generator = build_generator()
critic = build_critic()

# Initialize the Conditional GAN
cgan = ConditionalGAN(generator, critic)

# Set up optimizers and loss function
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Compile the Conditional GAN
cgan.compile(d_optimizer, g_optimizer, loss_fn)

# Training loop
for epoch in range(100):  # Train for 100 epochs.
    # Here I use dummy data for the example. You should load a proper dataset.
    real_images = np.random.rand(32, 28, 28, 1)  # Dummy real images (random values).
    labels = np.random.randint(0, 2, 32)  # Random labels (0 or 1 for blond or non-blond).

    # Perform one training step
    results = cgan.train_step(real_images, labels)
    print(f"Epoch: {epoch}, D Loss: {results['d_loss']}, G Loss: {results['g_loss']}")

    # Visualize generated images every 10 epochs
    if epoch % 10 == 0:
        plot_generated_images(epoch, generator, examples=10, noise_dim=100)
