from pathlib import Path

from model_training.data_generation import DataGeneration
from model_training.visual_decoders import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device =", device)

data_id = ""
model_id = "vae"

network = VAE_CNN()

# Load data
X, Y, data_range = DataGeneration().load_data(data_id)

# Make a directory for the model to be trained
Path(network.SAVE_PATH + "/" + model_id + "/").mkdir(parents=True, exist_ok=True)

# Save a copy of the data range of the data at the model location
np.save(network.SAVE_PATH + "/" + model_id + "/data_range" + model_id, data_range)

# Train the network
network.train_net(network, X, Y, model_id, max_epochs=800, batch_size=128)

