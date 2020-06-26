from model_training.data_generation import DataGeneration
from model_training.visual_decoders import *
from unity import UnityContainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device =", device)

editor_mode = 1
data_id = "s2nr"

data_gen = DataGeneration()

# Initialise Unity environment
unity = UnityContainer(editor_mode)
unity.initialise_environment()

# Generate and save the data
data_gen.generate_data(unity, data_id)

# Close the Unity environment
unity.close()
