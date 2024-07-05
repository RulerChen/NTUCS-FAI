import os
import numpy as np
import torch
from PIL import Image
from src.pca import PCA
from src.autoencoder import Autoencoder, DenoisingAutoencoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import torch.nn as nn

# TODO: change to your data path
DATA_PATH = './data'
# set random seed
np.random.seed(0)
"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


def read_image():
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    file_path = './data/subject_05_17.png'  # TODO: change to your path
    img = Image.open(file_path).convert("L")
    img_array = np.array(img)
    img_vector = img_array.flatten()
    img_vector = img_vector/255.0
    return np.array(img_vector, dtype='float')


def load_data(split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    data_path = DATA_PATH+'/'+split
    files = os.listdir(data_path)
    image_vectors = []
    label_vectors = []

    for f in files:
        # Read the image using PIL
        img = Image.open(data_path + '/'+f).convert("L")
        f_name, f_type = os.path.splitext(f)
        label = int(f_name[-2:])-1
        label_vectors.append(label)

        # Convert the image to a numpy array
        img_array = np.array(img)

        # Reshape the image into a vector
        img_vector = img_array.flatten()
        img_vector = img_vector/255.0
        image_vectors.append(img_vector)

    return np.array(image_vectors), np.array(label_vectors)


def compute_acc(y_pred: np.ndarray, y_val: np.ndarray):
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    return np.sum(y_pred == y_val) / len(y_val)


def reconstruction_loss(img_vec: np.ndarray, img_vec_reconstructed: np.ndarray) -> float:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    return ((img_vec - img_vec_reconstructed)**2).mean()


def problem_a(pca):
    pca.problem_a()


def problem_b(autoencoder, deno_autoencoder):
    plt.plot(autoencoder.training_curve, 'r.--',
             linewidth=1, markersize=1, label='autoencoder')
    plt.plot(deno_autoencoder.training_curve, 'b.--',
             linewidth=1, markersize=1, label='deno_autoencoder')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend(loc='upper right')

    if not os.path.exists('image'):
        os.makedirs('image')

    plt.savefig('image/problem_b.png')
    plt.clf()


def problem_c(pca, autoencoder, deno_autoencoder, img_vec):
    img_reconstruct_pca = pca.reconstruct(img_vec)
    img_reconstruct_ae = autoencoder.reconstruct(
        torch.tensor(img_vec, dtype=torch.float32))
    img_reconstruct_deno_ae = deno_autoencoder.reconstruct(
        torch.tensor(img_vec, dtype=torch.float32))

    reconstruction_loss_pca = reconstruction_loss(img_vec, img_reconstruct_pca)
    reconstruction_loss_ae = reconstruction_loss(img_vec, img_reconstruct_ae)
    reconstruction_loss_deno_ae = reconstruction_loss(
        img_vec, img_reconstruct_deno_ae)

    plt.figure(figsize=(15, 10))

    plt.subplot(321)
    plt.imshow(np.reshape(img_vec, (61, 80)))
    plt.title('Original Image')
    plt.subplot(322)
    plt.imshow(np.reshape(img_reconstruct_pca, (61, 80)))
    plt.title('PCA Loss: {:.6f}'.format(reconstruction_loss_pca))
    plt.subplot(323)
    plt.imshow(np.reshape(img_vec, (61, 80)))
    plt.title('Original Image')
    plt.subplot(324)
    plt.imshow(np.reshape(img_reconstruct_ae, (61, 80)))
    plt.title('Autoencoder Loss: {:.6f}'.format(reconstruction_loss_ae))
    plt.subplot(325)
    plt.imshow(np.reshape(img_vec, (61, 80)))
    plt.title('Original Image')
    plt.subplot(326)
    plt.imshow(np.reshape(img_reconstruct_deno_ae, (61, 80)))
    plt.title('DenoisingAutoencoder Loss: {:.6f}'.format(
        reconstruction_loss_deno_ae))

    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    if not os.path.exists('image'):
        os.makedirs('image')

    plt.savefig('image/problem_c.png')
    plt.clf()


def problem_d(X, img_vec):
    input_dim, encoding_dim = 4880, 488
    deno_autoencoder1, deno_autoencoder2 = DenoisingAutoencoder(
        input_dim=input_dim, encoding_dim=encoding_dim), DenoisingAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
    deno_autoencoder1.encoder = nn.Sequential(
        nn.Linear(input_dim, encoding_dim*2),
        nn.ReLU(),
        nn.Linear(encoding_dim*2, encoding_dim),
        nn.ReLU()
    )
    deno_autoencoder1.decoder = nn.Sequential(
        nn.Linear(encoding_dim, encoding_dim*2),
        nn.ReLU(),
        nn.Linear(encoding_dim*2, input_dim),
        nn.Tanh()
    )
    deno_autoencoder2.encoder = nn.Sequential(
        nn.Linear(input_dim, encoding_dim),
        nn.ReLU(),
        nn.Linear(encoding_dim, encoding_dim//2),
        nn.ReLU(),
        nn.Linear(encoding_dim//2, encoding_dim//4),
        nn.ReLU(),
    )
    deno_autoencoder2.decoder = nn.Sequential(
        nn.Linear(encoding_dim//4, encoding_dim//2),
        nn.ReLU(),
        nn.Linear(encoding_dim//2, encoding_dim),
        nn.ReLU(),
        nn.Linear(encoding_dim, input_dim),
        nn.Tanh()
    )

    deno_autoencoder1.fit(X, epochs=500, batch_size=135, learning_rate=1e-4)
    deno_autoencoder2.fit(X, epochs=500, batch_size=135, learning_rate=1e-4)
    plt.plot(deno_autoencoder1.training_curve, 'r.--',
             linewidth=1, markersize=1, label='Method 1')
    plt.plot(deno_autoencoder2.training_curve, 'b.--',
             linewidth=1, markersize=1, label='Method 2')
    plt.legend(loc='upper right')

    plt.xlabel('epoch')
    plt.ylabel('MSE Loss')

    if not os.path.exists('image'):
        os.makedirs('image')

    plt.savefig('image/problem_d.png')

    plt.clf()

    img_reconstruct_deno_ae1 = deno_autoencoder1.reconstruct(
        torch.tensor(img_vec, dtype=torch.float32))
    img_reconstruct_deno_ae2 = deno_autoencoder2.reconstruct(
        torch.tensor(img_vec, dtype=torch.float32))
    reconstruction_loss_deno_ae1 = reconstruction_loss(
        img_vec, img_reconstruct_deno_ae1)
    reconstruction_loss_deno_ae2 = reconstruction_loss(
        img_vec, img_reconstruct_deno_ae2)
    print(reconstruction_loss_deno_ae1, reconstruction_loss_deno_ae2)


def problem_e(X):
    deno_autoencoder1, deno_autoencoder2 = DenoisingAutoencoder(
        input_dim=4880, encoding_dim=488), DenoisingAutoencoder(input_dim=4880, encoding_dim=488)
    deno_autoencoder1.fit(X, epochs=500, batch_size=135,
                          learning_rate=1e-4, opti=torch.optim.AdamW)
    deno_autoencoder2.fit(X, epochs=500, batch_size=135,
                          learning_rate=5e-2, opti=torch.optim.SGD)
    plt.plot(deno_autoencoder1.training_curve, 'r.--',
             linewidth=1, markersize=1, label='AdamW')
    plt.plot(deno_autoencoder2.training_curve, 'b.--',
             linewidth=1, markersize=1, label='SGD')

    plt.legend(loc='upper right')
    plt.title('deno autoencoder')
    plt.xlabel('epoch')
    plt.ylabel('MSE Loss')

    if not os.path.exists('image'):
        os.makedirs('image')

    plt.savefig('image/problem_e.png')

    plt.clf()


def main():
    print("Loading data...")
    X_train, y_train = load_data("train")
    X_val, y_val = load_data("val")
    # Prepare data
    # PCA
    pca = PCA(n_components=40)
    print("PCA Training Start...")
    pca.fit(X_train)

    # Autoencoder
    autoencoder = Autoencoder(input_dim=4880, encoding_dim=488)
    print("Autoencoder Training Start...")
    autoencoder.fit(X_train, epochs=500, batch_size=135)

    # # DenoisingAutoencoder
    deno_autoencoder = DenoisingAutoencoder(input_dim=4880, encoding_dim=488)
    print("DenoisingAutoencoder Training Start...")
    deno_autoencoder.fit(X_train, epochs=500, batch_size=135)

    # Feature Transform: PCA
    print('Feature Transformation')
    X_train_transformed_pca = pca.transform(X_train)
    X_val_transformed_pca = pca.transform(X_val)

    # Feature Transform: Autoencoder
    X_train_transformed_ae = autoencoder.transform(X_train)
    X_val_transformed_ae = autoencoder.transform(X_val)

    # # Feature Transform: Autoencoder
    X_train_transformed_deno_ae = deno_autoencoder.transform(X_train)
    X_val_transformed_deno_ae = deno_autoencoder.transform(X_val)

    # Logistic Regression
    # create a logistic regression model
    clf_pca = LogisticRegression(max_iter=10000, random_state=0)
    clf_ae = LogisticRegression(max_iter=10000, random_state=0)
    clf_deno_ae = LogisticRegression(max_iter=10000, random_state=0)

    # fit the model to the data
    print("Logistic Regression Training Start...")
    clf_pca.fit(X_train_transformed_pca, y_train)
    clf_ae.fit(X_train_transformed_ae, y_train)
    clf_deno_ae.fit(X_train_transformed_deno_ae, y_train)

    # make predictions on new data
    y_pred_pca = clf_pca.predict(X_val_transformed_pca)
    y_pred_ae = clf_ae.predict(X_val_transformed_ae)
    y_pred_deno_ae = clf_deno_ae.predict(X_val_transformed_deno_ae)
    print(f"Acc from PCA: {compute_acc(y_pred_pca, y_val)}")
    print(f"Acc from Autoencoder: {compute_acc(y_pred_ae, y_val)}")
    print(
        f"Acc from DenoisingAutoencoder: {compute_acc(y_pred_deno_ae, y_val)}")

    # Reconstruct Image: subject05_17.png
    img_vec = read_image()
    img_reconstruct_pca = pca.reconstruct(img_vec)
    img_reconstruct_ae = autoencoder.reconstruct(
        torch.tensor(img_vec, dtype=torch.float32))
    img_reconstruct_deno_ae = deno_autoencoder.reconstruct(
        torch.tensor(img_vec, dtype=torch.float32))

    reconstruction_loss_pca = reconstruction_loss(img_vec, img_reconstruct_pca)
    reconstruction_loss_ae = reconstruction_loss(img_vec, img_reconstruct_ae)
    reconstruction_loss_deno_ae = reconstruction_loss(
        img_vec, img_reconstruct_deno_ae)

    print(f"Reconstruction Loss with PCA: {reconstruction_loss_pca}")
    print(f"Reconstruction Loss with Autoencoder: {reconstruction_loss_ae}")
    print(
        f"Reconstruction Loss with DenoisingAutoencoder: {reconstruction_loss_deno_ae}")

    problem_a(pca)
    problem_b(autoencoder, deno_autoencoder)
    problem_c(pca, autoencoder, deno_autoencoder, img_vec)
    problem_d(X_train, img_vec)
    problem_e(X_train)


if __name__ == "__main__":
    main()
