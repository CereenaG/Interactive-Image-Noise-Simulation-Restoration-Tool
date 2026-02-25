import matplotlib.pyplot as plt


def show_results(title,
                 original,
                 noisy,
                 mean_img,
                 gaussian_img,
                 median_img):

    plt.figure(figsize=(12, 8))
    plt.suptitle(title, fontsize=16)

    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Noisy Image")
    plt.imshow(noisy)
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("Mean Filter")
    plt.imshow(mean_img)
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("Gaussian Filter")
    plt.imshow(gaussian_img)
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("Median Filter")
    plt.imshow(median_img)
    plt.axis("off")

    plt.tight_layout()
    plt.show()