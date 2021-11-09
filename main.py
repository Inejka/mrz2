import network
import images_manager as im
import numpy as np

if __name__ == '__main__':
    print(im.images_manager.get_max(1000))
    n = network.network(im.images_manager.get_images(400, 1000), mode="projection")
    n.learn_images()
    res = []
    for i in range(100):
        correct_ans, image = im.images_manager.get_corrupted_image(n.get_images(), 40)
        ans, iter = n.recognize(image, correct_ans)
        res.append(correct_ans == ans)
        print(res[i], "   ", iter)
    print(np.sum(res))
