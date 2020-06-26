import csv

import numpy as np
import pandas as pd

from utility_functions import forward_kinematics


class CSVLogger:
    """
    Class containing the functions used by the FepAgent class to log hyperparameters and iteration data to a CSV file
    """
    def __init__(self, log_id):
        """
        Initialise the CSV logger
        :param log_id: name of the log file (the file will be saved at /model_operation/operation_logs/[log_id].csv)
        """
        self.log_id = log_id

    def write_header(self, fep_agent):
        """
        Write the header of the log file, containing the parameters of the fep_agent and the column headers for the
        iteration data
        :param fep_agent: FepAgent object to write the header for
        """
        with open('./operation_logs/' + self.log_id + '.csv', mode='w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['sigma_v_mu', 'sigma_v_a', 'sigma_p', 'sigma_mu', 'attractor_active', 'beta',
                                 'sp_active', 'sp_noise_variance'])
            filewriter.writerow([fep_agent.sigma_v_mu, fep_agent.sigma_v_a, fep_agent.sigma_p, fep_agent.sigma_mu,
                                 fep_agent.attractor_active, fep_agent.gamma, fep_agent.sp_active,
                                 fep_agent.sp_noise_variance])
            filewriter.writerow(['Iteration', 'A_Shoulder', 'A_Elbow', 'A_dot_Shoulder', 'A_dot_Elbow', 'mu_Shoulder',
                                 'mu_Elbow', 'sp_Shoulder', 'sp_Elbow', 'Ev attr', 'beta', 'last_vt', 'last_tt',
                                 'cartesian_distance', 'rubb_Shoulder', 'rubb_Elbow'])

    def write_iteration(self, fep_agent, i):
        """
        Write the iteration state data to the CSV file
        :param fep_agent: fep_agent to write the iteration state for
        :param i: current iteration index
        """
        with open('./operation_logs/' + self.log_id + '.csv', mode='a', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow([i, fep_agent.a[0, 0], fep_agent.a[0, 1], fep_agent.a_dot[0, 0], fep_agent.a_dot[0, 1],
                                 fep_agent.mu[0, 0], fep_agent.mu[0, 1], fep_agent.s_p[0, 0], fep_agent.s_p[0, 1],
                                 fep_agent.attr_error_tracker, fep_agent.gamma, fep_agent.last_tv, fep_agent.last_tt,
                                 fep_agent.env.get_cartesian_distance(),
                                 fep_agent.env.get_rubber_joint_observation()[0, 0],
                                 fep_agent.env.get_rubber_joint_observation()[0, 1]])

    @staticmethod
    def import_log(log_id, n, length, columns):
        """
        Import CSV operation_logs of multiple runs into a numpy array
        :param log_id: log id of the operation_logs to import (files need to have format log_id[I_RUN].csv
                       where [I_RUN] is an integer denoting the run number (starting at 0).
        :param n: number of runs to import
        :param length: length of the runs (needs to be the same for each run)
        :param columns: list containing the names of the colums to import
        :return: numpy array of shape (n, length, len(columns))
        """
        data = np.zeros((n, length, len(columns)))

        for i in range(n):
            df = pd.read_csv('../model_operation/operation_logs/' + log_id + str(i) + '.csv', header=2)
            loaded_data = []
            for column in columns:
                if df[column].dtype != 'float64':
                    loaded_data.append(df[column].map(lambda x: x.lstrip('[').rstrip(']')).astype('float64'))
                else:
                    loaded_data.append(df[column])
            data[i] = np.array(loaded_data).T

        return data

    @staticmethod
    def get_rhi_data(model, n, length, variable):
        """
        Easy import function for the RHI data that takes into account name formatting
        :param model: model name
        :param n: number of runs to import
        :param length: length of the runs (needs to be the same for each run)
        :param variable: the variable to retrieve (e.g. mu or a)
        :return: numpy array of shape (3, 2, 3, n, length) with the following dimension meaning:
                (condition, stimulation, joint, n, length) where joint is structured as
                (shoulder, elbow, end-effector)
        """
        conditions = ["left", "center", "right"]
        stimulation_types = ["async", "sync"]

        data = np.zeros((3, 2, 3, n, length))
        for i_c in range(len(conditions)):
            for i_s in range(len(stimulation_types)):
                imported_data = CSVLogger.import_log(log_id="rhi_results/" + model + conditions[i_c] + "rhi" +
                                                            stimulation_types[i_s] + "0n", n=5, length=length,
                                                     columns=[variable+"_Shoulder", variable+"_Elbow"])
                data[i_c, i_s, :2] = np.transpose(imported_data, (2, 0, 1))
                data[i_c, i_s, 2] = forward_kinematics(data[i_c, i_s, 0], data[i_c, i_s, 1])

        return data
