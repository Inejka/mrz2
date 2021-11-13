"""
////////////////////////////////////////////////////////////////////////////////////
// Лабораторная работа 4 по дисциплине МРЗвИС
// Выполнена студентом группы 9217023
// БГУИР Павлов Даниил Иванович
// Вариант 2 - модель сети Хопфилда с непрерывным состоянием и
// дискретным временем в асинхронном режиме
// 10.11.2021
// Использованные материалы:
// https://numpy.org/doc/stable/index.html - методические материалы по numpy
// https://www.learnpython.org/ - методические материалы по python
// https://intuit.ru/studies/courses/61/61/lecture/20452?page=3 - описание сети
"""
import network
import images_manager as im
import numpy as np
from PIL import Image
import plot_maker as pm

if __name__ == '__main__':
    transform = {-1: [0, 0, 0], 0: [125, 125, 125], 1: [255, 255, 255]}
    print(transform)
    print(im.images_manager.get_max(1000))
    n = network.network(im.images_manager.get_images(10, 250), mode="projection")
    n.learn_images()
    images = n.get_images()
    # pm.plot_maker.iter_from_percent(n, 5, 40, 2, 20)
    # pm.plot_maker.iter_from_pixel_count(250, 1000, 50, 20)
    # pm.plot_maker.iter_from_percent_of_image(20, 40, 1, 20)
    res = []
    for i in range(len(images)):
        image = im.images_manager.corrupt_image(images[i], 40)

        ans, iter = n.recognize(image, None)
        test = []
        for pixel in image:
            if (pixel == -1):
                test.append([0, 0, 0])
            if (pixel == 0):
                test.append([124, 124, 124])
            if (pixel == 1):
                test.append([255, 255, 255])
        test = np.uint8(test)
        print(test)
        test = Image.fromarray(np.array(test).reshape(25, 10, 3)).resize((400,160), Image.ANTIALIAS).show()
        # print(images[i])
        res.append(i == ans)
        print(res[i], "   ", iter)
    print(np.sum(res))
    # params count - 50 , size - 250 , percent - 25
