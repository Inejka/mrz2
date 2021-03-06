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
import numpy as np
import matplotlib.pyplot as mpl
import images_manager as im
import network


class plot_maker:
    @staticmethod
    def iter_from_percent(net, start_point, end_point, step, tests):
        iter_average = []
        images = net.get_images()
        for i in range(start_point, end_point, step):
            print(i, "/", end_point)
            iters = []
            for j in images:
                ans, iter = net.recognize(im.images_manager.corrupt_image(j, i), None)
                iters.append(iter)
            iter_average.append(np.average(iters))
        print(iter_average)
        mpl.xlabel('Процент зашумления')
        mpl.ylabel('Количество итераций')
        mpl.plot(range(start_point, end_point, step), iter_average)
        mpl.show()

    @staticmethod
    def iter_from_pixel_count(start_point, end_point, step, tests):
        iter_average = []
        for i in range(start_point, end_point, step):
            print(i, "/", end_point)
            net = network.network(im.images_manager.get_images(50, i), mode="projection")
            net.learn_images()
            images = net.get_images()
            iters = []
            for j in images:
                ans, iter = net.recognize(im.images_manager.corrupt_image(j, 25), None)
                iters.append(iter)
            iter_average.append(np.average(iters))
        print(iter_average)
        mpl.xlabel('Размер изображения')
        mpl.ylabel('Количество итераций')
        mpl.plot(range(start_point, end_point, step), iter_average)
        mpl.show()

    @staticmethod
    def iter_from_percent_of_image(start_point, end_point, step, tests):
        iter_average = []
        for i in range(start_point, end_point, step):
            print(i, "/", end_point)
            net = network.network(im.images_manager.get_images(int(250 * i / 100), 250), mode="projection")
            net.learn_images()
            images = net.get_images()
            iters = []
            for j in images:
                ans, iter = net.recognize(im.images_manager.corrupt_image(j, 25), None)
                iters.append(iter)
            iter_average.append(np.average(iters))
        print(iter_average)
        mpl.xlabel('Количество образов в процентах от размера')
        mpl.ylabel('Количество итераций')
        mpl.plot(range(start_point, end_point, step), iter_average)
        mpl.show()
