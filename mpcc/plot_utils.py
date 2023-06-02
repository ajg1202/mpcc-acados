import matplotlib.pyplot as plt


def plot_track(lane_center, lane_inner, lane_outer):

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal')
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 120])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    plt.plot(lane_center[:, 0], lane_center[:, 1],
             linestyle='--', color='orange', linewidth=1)
    plt.plot(lane_inner[:, 0], lane_inner[:, 1],
             linestyle='-', color='black', linewidth=1.5)
    plt.plot(lane_outer[:, 0], lane_outer[:, 1],
             linestyle='-', color='black', linewidth=1.5)

    plt.show()
