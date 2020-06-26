import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from csv_logger import CSVLogger
from utility_functions import *

font = {'size': 12}

matplotlib.rc('font', **font)


def line_and_ribbon_plot(means, stds):
    """
    Plot a line and ribbon plot using means and standard deviations over time.
    Saves a vector image copy to line_and_ribbon_plot.pdf
    :param means: array of mean values
    :param stds: array of standard deviations
    """
    fig, ax = plt.subplots(1)
    fig.set_figheight(4)
    fig.set_figwidth(8.5)

    colours = np.array([[0.7, 0.3, 0.3], [0.7, 0.3, 0.3],
                        [0.0, 0.3, 0.7], [0.0, 0.3, 0.7]])

    ax.plot([-10, 5000], [0, 0], c=[0.9, 0.9, 0.9], lw=0.5, zorder=0)
    for i in range(len(means)):
        print(colours[i])
        ax.plot(range(len(means[i][:, column])), means[i][:, column], lw=lw, linestyle="dashed" if i%2==0 else "solid", color=tuple(colours[i]+0.05))
        ax.fill_between(range(len(means[i][:, column])), means[i][:, column] + stds[i][:, column],
                        means[i][:, column] - stds[i][:, column],
                        facecolor=colours[i],
                        edgecolor=tuple(colours[i]+0.05),
                        linestyle="dashed" if i % 2 == 0 else "solid",
                        alpha=0.3 if i % 2 == 0 else 0.3)

    # ax.set_title("Shift in shoulder angle belief $\mu$ for the left \n condition under asynchronous stimulation")
    red_dash, = ax.plot([0, 1], [-50, -50], lw=lw, linestyle="dashed", color=tuple(colours[0] + 0.05))
    blue_dash, = ax.plot([0, 1], [-50, -50], lw=lw, linestyle="dashed", color=tuple(colours[2] + 0.05))

    red_solid, = ax.plot([0, 1], [-50, -50], lw=lw, linestyle="solid", color=tuple(colours[0] + 0.05))
    blue_solid, = ax.plot([0, 1], [-50, -50], lw=lw, linestyle="solid", color=tuple(colours[2] + 0.05))

    vae_patch = mpatches.Patch(color=colours[0] + 0.05, label='VAE')
    conv_patch = mpatches.Patch(color=colours[2] + 0.05, label='ConvDecoder')
    ax.legend([vae_patch, conv_patch, (red_solid, blue_solid), (red_dash, blue_dash)],
              ['VAE', 'ConvDecoder', 'Far', 'Close'],
              numpoints=1,
              handler_map={tuple: HandlerTuple(ndivide=None)},
              ncol=2, fontsize="medium", handletextpad=0.5, labelspacing=0.5, columnspacing=0.5, loc="upper right")
    ax.set_ylabel("$L^2$ distance from goal in $m$")
    #ax.set_ylabel("MSE between $s_v$ and $s_{v_{attr}}$")
    ax.set_xlim((0, 5000))
    ax.set_xlabel('Iteration')
    ax.set_ylim((-0.007, 0.20))
    #ax.set_ylim((-0.004, 0.065))

    fig.savefig('line_and_ribbon_plot.pdf', format='pdf')
    plt.show()


def mean_std_plot(vae_data, conv_data, sync, joint):
    """
    Plot a dot plot of means and standard deviations of the rubber-hand illusion data
    :param vae_data: data of the variational autoencoder
    :param conv_data: data of the convolutional decoder
    :param sync: whether to plot the synchronous or asynchronous condition
    :param joint: which joint to plot (0 for shoulder, 1 for elbow, 2 for end-effector)
    Saves a vector image copy to mean_std_plot.pdf
    """
    fig, ax = plt.subplots(1)
    fig.set_figwidth(3.5)
    fig.set_figheight(3)

    c = np.array([[[0.3, 0.3, 0.8]],
                  [[0.3, 0.8, 0.3]],
                  [[0.8, 0.3, 0.3]]])

    ax.plot([-10, 10], [0, 0], c=[0.9, 0.9, 0.9], lw=0.5, zorder=0)

    for i_c in range(3):
        mean = np.mean(vae_data[i_c, sync, joint])
        print("vae", sync, joint, "cond", i_c, "mu", mean)
        std = np.std(vae_data[i_c, sync, joint])
        print("vae", sync, joint, "cond", i_c, "std", std)
        ax.scatter(i_c+1, mean, c=c[i_c], s=75, edgecolors="None", label="VAE" if i_c == 1 else " ")
        ax.vlines(i_c+1, mean-std, mean+std, colors=c[i_c], lw=1.5)

    c = np.array([[[0.76, 0.89, 0.96]],
                  [[0.62, 0.96, 0.72]],
                  [[0.95, 0.67, 0.64]]])

    for i_c in range(3):
        mean = np.mean(conv_data[i_c, sync, joint])
        print("conv", sync, joint, "cond", i_c, "mu", mean)
        std = np.std(conv_data[i_c, sync, joint])
        print("conv", sync, joint, "cond", i_c, "std", std)
        ax.scatter(i_c+1+0.2, mean, c=c[i_c], s=50, edgecolors="None", label ="ConvDecoder" if i_c == 1 else " ")
        ax.vlines(i_c+1+0.2, mean-std, mean+std, colors=c[i_c], linestyles="dashed", lw=1.5)


    ax.set_xlim((0.5, 3.5))
    #ax.set_ylim((-0.48, 0.48))
    ax.set_ylim((-1.0, 1.0))
    #ax.set_yticks(np.arange(-1.0, 1.0, 0.2))
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['L', 'C', 'R'])
    ax.legend(loc='upper left', ncol=2, fontsize="small", handletextpad = 0.0, labelspacing=0.0, columnspacing=0.5)
    #ax.set_ylabel("Attempted acceleration ($m s^{-2}$)")
    ax.set_ylabel("Perceptual end effector drift ($cm$)")
    #fig.axes[0].get_yaxis().set_visible(False)
    fig.subplots_adjust(left=0.2357)
    fig.savefig('mean_std_plot.pdf', format='pdf')
    plt.show()


def jacobians_plot(scale=25):
    """
    Plot the Jacobians of the both joints for both models
    :param scale: defines the extreme values the plot uses to adjust its colour scale
    Saves a vector image copy to jacobians.pdf
    """
    fig, ax = plt.subplots(1, 4)
    fig.set_figwidth(16)
    fig.set_figheight(4)

    titles = ["VAE Shoulder", "ConvDecoder Shoulder", "VAE Elbow", "ConvDecoder Elbow"]
    jacobians = ["jacobians/jac_vae.npy", "jacobians/jac_conv_decoder.npy"]

    for i in range(4):
        jac = np.squeeze(np.load(jacobians[i % 2]))
        ax[i].imshow(jac[:, :, 0 if i <= 1 else 1].T, cmap='bwr', vmin=-scale, vmax=scale)
        ax[i].set_xlim((0, 135))
        ax[i].set_xticks(())
        ax[i].set_yticks(())
        ax[i].set_ylim((255, 120))
        ax[i].set_title(titles[i], **{'fontname': 'Times New Roman'})

    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("jacobians.pdf", bbox_inches='tight',
                pad_inches=0)
    plt.show()


def mean_and_std(data, axis=0):
    """Compute the mean and standard deviation over a specified axis"""
    return np.average(data, axis=axis), np.std(data, axis=axis)


model = "vae"

data_1 = CSVLogger.import_log(log_id="reach_results/" + model + "reach" + "close" + "0n", n=40, length=5000,
                              columns=["cartesian_distance", "Ev attr", "A_Shoulder", "A_Elbow", "rubb_Shoulder", "rubb_Elbow"])

mean_1, std_1 = mean_and_std(data_1)

data_2 = CSVLogger.import_log(log_id="reach_results/" + model + "reach" + "far" + "0n", n=40, length=5000,
                              columns=["cartesian_distance", "Ev attr","A_Shoulder", "A_Elbow", "rubb_Shoulder", "rubb_Elbow"])


mean_2, std_2 = mean_and_std(data_2)


column = 0
lw = 1


model = "s2nr2u"

data_3 = CSVLogger.import_log(log_id="reach_results/" + model + "reach" + "close" + "0n", n=40, length=5000,
                              columns=["cartesian_distance", "Ev attr", "A_Shoulder", "A_Elbow", "rubb_Shoulder", "rubb_Elbow"])

mean_3, std_3 = mean_and_std(data_3)

data_4 = CSVLogger.import_log(log_id="reach_results/" + model + "reach" + "far" + "0n", n=40, length=5000,
                              columns=["cartesian_distance", "Ev attr", "A_Shoulder", "A_Elbow", "rubb_Shoulder", "rubb_Elbow"])


mean_4, std_4 = mean_and_std(data_4)

jacobians_plot()



