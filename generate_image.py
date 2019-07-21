import numpy as np
from tqdm import tqdm
from keras.initializers import RandomNormal
from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def preprocess(x):
    return (x/255)*2-1

def deprocess(x):
    return np.uint8((x+1)/2*255)

def make_generator(input_size, leaky_alpha, init_stddev):
    # generates images in size (32,32,3)
    return Sequential([
        Dense(4*4*512, input_shape=(input_size,),
              kernel_initializer=RandomNormal(stddev=init_stddev)),
        Reshape(target_shape=(4, 4, 512)),
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),
        Conv2DTranspose(256, kernel_size=5, strides=2, padding='same',
                        kernel_initializer=RandomNormal(stddev=init_stddev)), # 8x8
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),
        Conv2DTranspose(128, kernel_size=5, strides=2, padding='same',
                        kernel_initializer=RandomNormal(stddev=init_stddev)), # 16x16
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),
        Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                        kernel_initializer=RandomNormal(stddev=init_stddev)), # 32x32
        Activation('tanh')
    ])

def make_discriminator(leaky_alpha, init_stddev):
    # classifies images in (64,64,3)
    return Sequential([
        Conv2D(64, kernel_size=5, strides=2, padding='same',
               kernel_initializer=RandomNormal(stddev=init_stddev),    # 16x16
               input_shape=(32, 32, 3)),
        LeakyReLU(alpha=leaky_alpha),
        Conv2D(128, kernel_size=5, strides=2, padding='same',
               kernel_initializer=RandomNormal(stddev=init_stddev)),   # 8x8
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),
        Conv2D(256, kernel_size=5, strides=2, padding='same',
               kernel_initializer=RandomNormal(stddev=init_stddev)),   # 4x4
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),
        Flatten(),
        Dense(1, kernel_initializer=RandomNormal(stddev=init_stddev)),
        Activation('sigmoid')
    ])


# beta_1 is the exponential decay rate for the 1st moment estimates in Adam optimizer
def make_DCGAN(sample_size,
               g_learning_rate,
               g_beta_1,
               d_learning_rate,
               d_beta_1,
               leaky_alpha,
               init_std):
    # generator
    generator = make_generator(sample_size, leaky_alpha, init_std)

    # discriminator
    discriminator = make_discriminator(leaky_alpha, init_std)
    discriminator.compile(optimizer=Adam(lr=d_learning_rate, beta_1=d_beta_1), loss='binary_crossentropy')

    # GAN
    gan = Sequential([generator, discriminator])
    gan.compile(optimizer=Adam(lr=g_learning_rate, beta_1=g_beta_1), loss='binary_crossentropy')

    return gan, generator, discriminator


def make_latent_samples(n_samples, sample_size):
    # return np.random.uniform(-1, 1, size=(n_samples, sample_size))
    return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))


def make_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable


def make_labels(size):
    return np.ones([size, 1]), np.zeros([size, 1])


def show_losses(losses):
    losses = np.array(losses)

    fig, ax = plt.subplots()
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Validation Losses")
    plt.legend()
    plt.show()


def show_images(generated_images):
    n_images = len(generated_images)
    cols = 10
    rows = n_images // cols

    plt.figure(figsize=(10, 8))
    for i in range(n_images):
        img = deprocess(generated_images[i])
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


def train(
        training_set,
        test_set,
        g_learning_rate,  # learning rate for the generator
        g_beta_1,  # the exponential decay rate for the 1st moment estimates in Adam optimizer
        d_learning_rate,  # learning rate for the discriminator
        d_beta_1,  # the exponential decay rate for the 1st moment estimates in Adam optimizer
        leaky_alpha,
        init_std,
        smooth=0.1,
        sample_size=100,  # latent sample size (i.e. 100 random numbers)
        epochs=3,
        batch_size=128,  # train batch size
        eval_size=16,  # evaluate size
        show_details=True
        ):
    # labels for the batch size and the test size
    y_train_real, y_train_fake = make_labels(batch_size)
    y_eval_real, y_eval_fake = make_labels(eval_size)

    # create a GAN, a generator and a discriminator
    gan, generator, discriminator = make_DCGAN(
        sample_size,
        g_learning_rate,
        g_beta_1,
        d_learning_rate,
        d_beta_1,
        leaky_alpha,
        init_std)

    losses = []
    for e in range(epochs):
        for i in tqdm(range(len(training_set) // batch_size)):
            # real CelebA images
            X_batch = training_set[i * batch_size:(i + 1) * batch_size]
            X_batch_real = np.array([preprocess(filename) for filename in X_batch])

            # latent samples and the generated digit images
            latent_samples = make_latent_samples(batch_size, sample_size)
            X_batch_fake = generator.predict_on_batch(latent_samples)

            # train the discriminator to detect real and fake images
            make_trainable(discriminator, True)
            discriminator.train_on_batch(X_batch_real, y_train_real * (1 - smooth))
            discriminator.train_on_batch(X_batch_fake, y_train_fake)

            # train the generator via GAN
            make_trainable(discriminator, False)
            gan.train_on_batch(latent_samples, y_train_real)

        # evaluate
        X_eval = test_set[np.random.choice(len(test_set), eval_size, replace=False)]
        X_eval_real = np.array([preprocess(filename) for filename in X_eval])

        latent_samples = make_latent_samples(eval_size, sample_size)
        X_eval_fake = generator.predict_on_batch(latent_samples)

        d_loss = discriminator.test_on_batch(X_eval_real, y_eval_real)
        d_loss += discriminator.test_on_batch(X_eval_fake, y_eval_fake)
        g_loss = gan.test_on_batch(latent_samples, y_eval_real)  # we want the fake to be realistic!

        losses.append((d_loss, g_loss))

        print("Epoch: {:>3}/{} Discriminator Loss: {:>6.4f} Generator Loss: {:>6.4f}".format(
            e + 1, epochs, d_loss, g_loss))
        show_images(X_eval_fake[:10])


    # show the result
    if show_details:
        show_losses(losses)

    return generator.predict(make_latent_samples(10000, sample_size))
