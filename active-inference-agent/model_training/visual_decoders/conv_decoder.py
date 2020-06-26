import os
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utility_functions import shuffle_unison
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ConvolutionalDecoder(nn.Module):

    SAVE_PATH = os.path.join(os.path.dirname(__file__), "trained_conv_decoder")

    def __init__(self):
        super(ConvolutionalDecoder, self).__init__()

        self.fc1 = nn.Linear(2, 1024)
        self.fc2 = nn.Linear(1024, 8 * 8 * 128)

        self.upconv1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.upconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.upconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.upconv4 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.drop = nn.Dropout(p=0.1)

        self.upconv5 = nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.to(device)

    def visual_prediction(self, mu):
        """
        Get visual prediction for a set of joint angles
        :param mu: joint angle belief to generate prediction for
        :return: visual prediction
        """
        return self.forward(mu)

    def forward(self, x):
        """
        Perform forward pass through the network
        :param x: input
        :return: network output
        """
        # Two fully connected layers of neurons:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # Reshaping the output of the fully connected layer so that it is compatible with the conv layers
        x = x.view(-1, 128, 8, 8)

        # Upsampling using the deconvolutional layers:
        x = self.relu(self.upconv1(x))
        x = self.relu(self.conv1(x))

        x = self.relu(self.upconv2(x))
        x = self.relu(self.conv2(x))

        x = self.relu(self.upconv3(x))
        x = self.relu(self.conv3(x))

        x = self.relu(self.upconv4(x))

        x = self.relu(self.conv4(x))

        # Squeezing the output to 0-1 range:
        x = self.sigmoid(self.upconv5(x))

        return x

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
        net.to(device)

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.10, random_state=11)

        Y_train = Y_train[:, np.newaxis, :, :]
        Y_val = Y_val[:, np.newaxis, :, :]

        optimizer = optim.Adam(net.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.95)

        criterion = nn.MSELoss()

        epoch_loss = []
        val_loss = []
        batch_loss = []

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plt.ion()
        fig.show()
        plt.pause(0.001)

        for epoch in range(max_epochs):
            cur_batch_loss = []
            cur_val_loss = []
            x, y = shuffle_unison(X_train, Y_train)
            for i in range(X_train.shape[0] // batch_size):
                loss = ConvolutionalDecoder.run_batch(i, x, y, True, net, optimizer, criterion, batch_size, device)
                cur_batch_loss = np.append(cur_batch_loss, [loss])
                batch_loss = np.append(batch_loss, [loss])

            scheduler.step()

            epoch_loss = np.append(epoch_loss, [np.mean(cur_batch_loss)])

            if epoch % 10 == 0 and epoch<max_epochs-1:
                x, y = shuffle_unison(X_val, Y_val)
                for i in range(X_val.shape[0]):
                    loss = ConvolutionalDecoder.run_batch(i, x, y, False, net, optimizer, criterion, 1, device)
                    cur_val_loss = np.append(cur_val_loss, [loss])
                val_loss = np.append(val_loss, [np.mean(cur_val_loss)])

                print('------ Epoch ', epoch, '--------LR:', scheduler.get_lr())
                print('Epoch loss:', epoch_loss[-1])
                print('Val loss:', val_loss[-1])
                torch.save(net.state_dict(), ConvolutionalDecoder.SAVE_PATH + "/" + network_id + "/trained_network" + network_id)

                ax.set_title("Loss")
                ax.plot(range(len(epoch_loss)), epoch_loss, label="Epoch loss")
                ax.plot(np.arange(len(val_loss))*10, val_loss, label="Validation loss")
                plt.pause(0.001)

        torch.save(net.state_dict(), ConvolutionalDecoder.SAVE_PATH + "/" + network_id + "/trained_network" + network_id)

    @staticmethod
    def run_batch(i, x, y, train, net, optimizer, criterion, batch_size, device):
        """
        Execute a training batch
        """
        input_x = torch.tensor(np.float32(x[i * batch_size: (i + 1) * batch_size]), device=device)
        target_y = torch.tensor(np.float32(y[i * batch_size: (i + 1) * batch_size]), device=device,
                                requires_grad=False)

        optimizer.zero_grad()
        output_y = net(input_x)
        loss = criterion(output_y, target_y)
        if train:
            loss.backward()
            optimizer.step()

        return loss.item()

    def load_from_file(self, model_id):
        """
        Load network from file
        :param model_id: save id to load from
        """
        self.load_state_dict(torch.load(os.path.join(self.SAVE_PATH, model_id + "/trained_network" + model_id)))
        self.eval()
