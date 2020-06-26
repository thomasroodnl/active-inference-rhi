import os
from pathlib import Path

import numpy as np

from utility_functions import min_max_norm_dr


class DataGeneration:
    """
    Class containing methods for generating a dataset from the unity environment.
    Note that data is collected by moving the real arm to a certain position and taking a snapshot,
    meaning that one needs to make sure to make the rubber arm invisible and the real arm visible to the camera.
    """
    OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "training_data")

    def __init__(self):
        self.shoulder_range = (-40, 40)
        self.elbow_range = (-50, 50)
        self.data_range = np.array([self.shoulder_range, self.elbow_range])
        self.interval = 1
        self.img_h = 256
        self.img_w = 256
        self.n_datapoints = 8000

    def generate_data(self, env, save_id):
        """
        Generate a dataset by randomly sampling n_datapoints in the given shoulder and elbow ranges
        and taking snapshots from the environment for these datapoints
        :param env: the environment object
        :param save_id: file id for saving the dataset on disk
        """
        x = np.zeros((self.n_datapoints, 1, 2))
        y = np.zeros((self.n_datapoints, self.img_h, self.img_w))

        rotations = np.stack((np.random.uniform(*self.shoulder_range, self.n_datapoints),
                              np.random.uniform(*self.elbow_range, self.n_datapoints))).T[:, np.newaxis, :]

        for i_rotate in range(len(rotations)):
            env.set_rotation(rotations[i_rotate])
            x[i_rotate] = min_max_norm_dr(env.get_joint_observation(), self.data_range)
            y[i_rotate] = np.squeeze(env.get_visual_observation())
            if i_rotate % 500 == 0:
                print("Generating data... "+str(round((i_rotate/self.n_datapoints)*100, 1))+"%")

        print("Generation complete: X"+str(x.shape), "Y"+str(y.shape))
        print("Writing data to disk, please wait...")

        Path(self.OUTPUT_PATH+"/"+save_id+"/").mkdir(parents=True, exist_ok=True)
        np.save(self.OUTPUT_PATH+"/"+save_id+"/x"+save_id, x)
        np.save(self.OUTPUT_PATH+"/"+save_id+"/y"+save_id, y)
        np.save(self.OUTPUT_PATH+"/"+save_id+"/data_range"+save_id, np.array([self.shoulder_range, self.elbow_range]))
        print("Data saved to "+self.OUTPUT_PATH+"/"+save_id+"/{x,y,data_range}"+save_id+".npy")

    def generate_data_ordered(self, env, save_id, dp_per_angle=90):
        """
        Generate data by taking dp_per_angle amount of datapoints for both angles (not randomly). The shoulder and
        elbow position are stored as two separated dimensions, allowing training algorithms to only train one
        joint at a time and clamp the other
        (as done in https://papers.nips.cc/paper/5851-deep-convolutional-inverse-graphics-network.pdf)
        :param env: environment object to collect the data from
        :param save_id: file id for saving the dataset on disk
        :param dp_per_angle: amount of datapoints
        """
        x = np.zeros((dp_per_angle, dp_per_angle, 1, 2))
        y = np.zeros((dp_per_angle, dp_per_angle, self.img_h, self.img_w))

        for r in range(dp_per_angle):
            for c in range(dp_per_angle):
                s_rotate = ((r - (dp_per_angle/2))/(dp_per_angle/2))*self.shoulder_range[1]
                e_rotate = ((c - (dp_per_angle/2))/(dp_per_angle/2))*self.elbow_range[1]
                env.set_rotation(np.array([[s_rotate, e_rotate]]))
                x[r, c] = min_max_norm_dr(env.get_joint_observation(), self.data_range)
                y[r, c] = np.squeeze(env.get_visual_observation())
                if r % (90//10) == 0:
                    print("Generating data... "+str(round(r/dp_per_angle*100, 1))+"%")

        print("Generation complete: X"+str(x.shape), "Y"+str(y.shape))
        print("Writing data to disk, please wait...")

        Path(self.OUTPUT_PATH+"/"+save_id+"/").mkdir(parents=True, exist_ok=True)
        np.save(self.OUTPUT_PATH+"/"+save_id+"/x"+save_id, x)
        np.save(self.OUTPUT_PATH+"/"+save_id+"/y"+save_id, y)
        np.save(self.OUTPUT_PATH+"/"+save_id+"/data_range"+save_id, np.array([self.shoulder_range, self.elbow_range]))
        print("Data saved to "+self.OUTPUT_PATH+"/"+save_id+"/{x,y,data_range}"+save_id+".npy")

    def load_data(self, save_id):
        """
        Load saved dataset from disk and return it
        :param save_id: the file id of the desired dataset
        :return: x(n_datapoints, 1, 2), y(n_datapoints, img_h, img_w) , data_range(2)
        """
        return np.load(self.OUTPUT_PATH+"/"+save_id+"/x"+save_id+".npy"), \
               np.load(self.OUTPUT_PATH+"/"+save_id+"/y"+save_id+".npy"), \
               np.load(self.OUTPUT_PATH+"/"+save_id+"/data_range"+save_id+".npy")
