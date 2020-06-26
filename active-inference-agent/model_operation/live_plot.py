import matplotlib.pyplot as plt
import numpy as np


class LivePlot:
    """
    Class containing the logic to create and update the live plot of the fep agent
    """
    def __init__(self, rows: int, columns: int):
        self.fig, self.ax = plt.subplots(rows, columns, figsize=(12, 9))
        plt.ion()
        self.fig.show()
        plt.pause(0.001)

    def set_line_plot(self, row: int, column: int, y, title: str):
        """
        Create a line plot at a certain location
        :param row: row index
        :param column: column index
        :param y: array of y-values
        :param title: plot title
        """
        self.ax[row, column].set_title(title)
        self.ax[row, column].plot(range(len(y)), y)

    def set_overlapping_image(self, row: int, column: int, img1: np.ndarray, img2: np.ndarray, cmap: str, title: str):
        """
        Create an overlapping image plot at a certain location
        :param row: row index
        :param column: column index
        :param img1: image 1
        :param img2: image 2 (the low alpha image)
        :param cmap: colour map
        :param title: plot title
        """
        self.ax[row, column].set_title(title)
        self.ax[row, column].imshow(np.squeeze(img1), cmap=cmap)
        self.ax[row, column].imshow(np.squeeze(img2), cmap=cmap, alpha=0.3)

    def update_live_plot(self, fep_agent):
        """
        Update the live plot
        :param fep_agent: the FepAgent object
        """
        self.set_overlapping_image(0, 0, img1=fep_agent.g_mu, img2=fep_agent.s_v[0, 0],
                                   cmap="Greys", title="Belief and perception")
        self.set_line_plot(0, 1, fep_agent.gamma_tracker, "Beta")

        for event in fep_agent.vt_tracker:
            self.ax[0, 1].axvline(x=event, c='b')
        for event in fep_agent.tt_tracker:
            self.ax[0, 1].axvline(x=event, c='cyan')

        self.set_line_plot(1, 0, fep_agent.mu_s_tracker, "Mu - shoulder")
        self.set_line_plot(1, 1, fep_agent.a_s_tracker, "a - shoulder")
        self.set_line_plot(2, 0, fep_agent.mu_e_tracker, "Mu - elbow")
        self.set_line_plot(2, 1, fep_agent.a_e_tracker, "a - elbow")

        self.refresh_plot()

    def refresh_plot(self):
        """
        Refresh the plot
        """
        plt.pause(0.001)
