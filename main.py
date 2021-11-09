import network
import images_manager as im
import numpy as np
import plot_maker as pm

if __name__ == '__main__':
    print(im.images_manager.get_max(1000))
    n = network.network(im.images_manager.get_images(50, 250), mode="projection")
    n.learn_images()
    pm.plot_maker.iter_from_percent(n, 5, 40, 2, 20)
    pm.plot_maker.iter_from_pixel_count(250, 1000, 50, 20)
    pm.plot_maker.iter_from_percent_of_image(20, 40, 1, 20)
    # res = []
    # for i in range(100):
    #    correct_ans, image = im.images_manager.get_corrupted_image(n.get_images(), 20)
    #    ans, iter = n.recognize(image, correct_ans)
    #    res.append(correct_ans == ans)
    #    print(res[i], "   ", iter)
    # print(np.sum(res))
    # params count - 50 , size - 250 , percent - 25
