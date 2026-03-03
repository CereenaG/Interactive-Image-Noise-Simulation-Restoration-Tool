import matplotlib.pyplot as plt
import cv2


def plot_histogram(image, title):
    fig, ax = plt.subplots()
    colors = ('b', 'g', 'r')

    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)

    ax.set_title(title)
    ax.set_xlim([0, 256])
    return fig