import numpy as np


class images_manager:
    @staticmethod
    def get_images(count, size):
        tmp = np.unique((np.random.randint(2, size=(count * 2, size)) - 1), axis=0)[0:count]
        tmp[tmp == 0] = 1
        return tmp

    @staticmethod
    def get_corrupted_image(images, percent):
        rand1 = np.random.choice(images.shape[0], replace=False)
        to_corrupt = images[rand1].copy()
        rands = np.random.choice(to_corrupt.size, size=int(to_corrupt.size * percent / 100), replace=False)
        for rand in rands:
            tmp = np.random.randint(-1, 1)
            while tmp == to_corrupt[rand]:
                tmp = np.random.randint(-1, 1)
            to_corrupt[rand] = tmp
        return rand1, to_corrupt

    @staticmethod
    def corrupt_image(image, percent):
        to_corrupt = image.copy()
        rands = np.random.choice(to_corrupt.size, size=int(to_corrupt.size * percent / 100), replace=False)
        for rand in rands:
            tmp = np.random.randint(-1, 1)
            while tmp == to_corrupt[rand]:
                tmp = np.random.randint(-1, 1)
            to_corrupt[rand] = tmp
        return to_corrupt

    @staticmethod
    def get_max(N):
        return np.divide(N, 2 * np.log2(N))
