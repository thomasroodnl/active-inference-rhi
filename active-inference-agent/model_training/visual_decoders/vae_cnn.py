from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import os
from torch.optim.lr_scheduler import StepLR
from utility_functions import shuffle_unison
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Adjusted from https://github.com/bhpfelix/Variational-Autoencoder-PyTorch/blob/master/src/vanila_vae.py
class VAE_CNN(nn.Module):
    SAVE_PATH = os.path.join(os.path.dirname(__file__), "trained_vae_cnn")

    def __init__(self):
        super(VAE_CNN, self).__init__()

        # Encoder
        self.e_conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.e_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.e_conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.e_conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.e_conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.e_conv5 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.e_fc1 = nn.Linear(8 * 8 * 128, 4096)
        self.e_fc2 = nn.Linear(4096, 1024)
        self.e_fc3 = nn.Linear(1024, 512)

        # Variational latent variable layers
        self.fc_mu = nn.Linear(512, 2)
        self.fc_logvar = nn.Linear(512, 2)

        # Decoder
        self.d_fc1 = nn.Linear(2, 1024)
        self.d_fc2 = nn.Linear(1024, 8 * 8 * 128)

        self.d_upconv1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.d_conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.d_upconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.d_conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.d_upconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.d_conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.d_upconv4 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.d_conv4 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.d_upconv5 = nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.to(device)

    def encode(self, x):
        """
        Run the encoder (first part of the forward pass)
        :param x: input image
        :return: mu and logvar vector
        """
        x = self.relu(self.e_pool(self.e_conv1(x)))
        x = self.relu(self.e_pool(self.e_conv2(x)))
        x = self.relu(self.e_pool(self.e_conv3(x)))
        x = self.relu(self.e_pool(self.e_conv4(x)))
        x = self.relu(self.e_pool(self.e_conv5(x)))

        # Reshaping the output of the fully conv layer so that it is compatible with the fc layers
        x = x.view(-1, 8 * 8 * 128)

        x = self.relu(self.e_fc1(x))
        x = self.relu(self.e_fc2(x))
        x = self.relu(self.e_fc3(x))

        # Return latent parameters
        return self.fc_mu(x), self.fc_logvar(x)

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Randomly sample z based on mu and logvar vectors
        :param mu: mu vector
        :param logvar: logvar vector
        :return: z
        """
        eps = torch.randn_like(logvar)
        return eps * torch.exp(0.5 * logvar) + mu

    def visual_prediction(self, mu):
        """
        Get visual prediction for a set of joint angles
        :param mu: joint angle belief to generate prediction for
        :return: visual prediction
        """
        return self.decode(mu)

    def decode(self, z):
        """
        Run the decoder (second part of the forward pass)
        :param z: latent variable vector z
        :return: output image
        """
        # Two fully connected layers of neurons:
        x = self.relu(self.d_fc1(z))
        x = self.relu(self.d_fc2(x))

        # Reshaping the output of the fully connected layer so that it is compatible with the conv layers
        x = x.view(-1, 128, 8, 8)

        # Upsampling using the deconvolutional layers:
        x = self.relu(self.d_upconv1(x))
        x = self.relu(self.d_conv1(x))

        x = self.relu(self.d_upconv2(x))
        x = self.relu(self.d_conv2(x))

        x = self.relu(self.d_upconv3(x))
        x = self.relu(self.d_conv3(x))

        x = self.relu(self.d_upconv4(x))
        x = self.relu(self.d_conv4(x))

        x = self.sigmoid(self.d_upconv5(x))
        return x

    def forward(self, x):
        """
        Perform forward pass through the network
        :param x: input
        :return: network output, mu and logvar vectors
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar

    @staticmethod
    def train_net(net, X, Y, network_id, max_epochs=600, batch_size=125):
        """
        Train the neural network
        :param net: the network object
        :param X: Input samples
        :param Y: Output samples
        :param network_id: network id for saving
        :param max_epochs: max number of epochs to train for
        :param batch_size: size of the mini-batches
        """
        torch.cuda.empty_cache()

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.10, random_state=58)

        Y_train = Y_train[:, np.newaxis, :, :]
        Y_val = Y_val[:, np.newaxis, :, :]

        optimizer = optim.Adam(net.parameters(), lr=0.001)  # 0.001
        scheduler = StepLR(optimizer, step_size=20, gamma=0.95)  # 0.95

        epoch_loss = []
        val_loss = []

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plt.ion()
        fig.show()
        plt.pause(0.001)

        for epoch in range(max_epochs):
            cur_batch_loss = []
            cur_val_loss = []
            x, y = shuffle_unison(X_train, Y_train)
            for i in range(X_train.shape[0] // batch_size):
                loss = VAE_CNN.run_batch(i, x, y, True, net, optimizer, batch_size)
                cur_batch_loss = np.append(cur_batch_loss, [loss])

            scheduler.step()

            epoch_loss = np.append(epoch_loss, [np.mean(cur_batch_loss)])

            if epoch % 10 == 0 and epoch < max_epochs - 1:
                x, y = shuffle_unison(X_val, Y_val)
                for i in range(X_val.shape[0]):
                    loss = VAE_CNN.run_batch(i, x, y, False, net, optimizer, 1)
                    cur_val_loss = np.append(cur_val_loss, [loss])
                val_loss = np.append(val_loss, [np.mean(cur_val_loss)])

                print('------ Epoch ', epoch, '--------LR:', scheduler.get_lr())
                print('Epoch loss:', epoch_loss[-1])
                print('Val loss:', val_loss[-1])
                torch.save(net.state_dict(),
                           VAE_CNN.SAVE_PATH + "/" + network_id + "/trained_network" + network_id + str(
                               epoch))

                ax.set_title("Loss")
                ax.set_yscale('log')
                ax.plot(range(len(epoch_loss)), epoch_loss, label="Epoch loss")
                ax.plot(np.arange(len(val_loss)) * 10, val_loss, label="Validation loss")
                plt.pause(0.001)

        torch.save(net.state_dict(),
                   VAE_CNN.SAVE_PATH + "/" + network_id + "/trained_network" + network_id + "final")

    @staticmethod
    def run_batch(i, x, y, train, net, optimizer, batch_size):
        """
        Execute a training batch
        """
        q = torch.tensor(x[i * batch_size: (i + 1) * batch_size], dtype=torch.float, device=device)
        input_y = torch.tensor(y[i * batch_size: (i + 1) * batch_size], dtype=torch.float, device=device)
        target_y = torch.tensor(y[i * batch_size: (i + 1) * batch_size], dtype=torch.float, device=device,
                                requires_grad=False)

        optimizer.zero_grad()
        predict_y, mu, logvar = net.forward(input_y)
        loss = VAE_CNN.loss_function(target_y, predict_y, q, mu, logvar)

        if train:
            loss.backward()
            optimizer.step()

        return loss.item()

    @staticmethod
    def loss_function(target_y, predict_y, q, mu, logvar):
        """
        Loss function of the VAE, based on the MSE of the images and regularisation term
        based on the Kullback-Leibler divergence
        :param target_y: target image
        :param predict_y: predicted image
        :param q: the ground truth joint angles
        :param mu: the latent mean joint angle vector
        :param logvar: the latent log variance joint angle vector
        :return: loss
        """
        criterion = nn.MSELoss()
        mse = criterion(predict_y, target_y)
        target_var = torch.zeros(logvar.shape, dtype=torch.float, device=device)
        target_var[:, :] = 0.001
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - torch.log(target_var) -
                                          (logvar.exp() + (q[:, 0] - mu) ** 2)/target_var, dim=1), dim=0)
        return mse + kld

    def load_from_file(self, model_id):
        """
        Load network from file
        :param model_id: save id to load from
        """
        self.load_state_dict(torch.load(os.path.join(self.SAVE_PATH, model_id+"/trained_network"+model_id)))
        self.eval()
