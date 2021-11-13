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


class network:
    def __init__(self, to_remember, mode):
        self.mode = mode
        self.images = to_remember
        self.W = np.zeros((to_remember[0].size, to_remember[0].size))

    def learn_images(self):
        if self.mode == "projection":
            for image in self.images:
                self.W += 1 / (image.reshape(1, image.size) @ image.reshape(image.size, 1) - (
                        image.reshape(1, image.size) @ self.W @ image.reshape(image.size, 1))) * \
                          (self.W @ image.reshape(image.size, 1) - image.reshape(image.size, 1)) @ \
                          (self.W @ image.reshape(image.size, 1) - image.reshape(image.size, 1)).transpose()
        if self.mode == "easy":
            for image in self.images:
                tmp = image.reshape(image.size, 1) @ image.reshape(1, image.size)
                self.W += tmp
                for i in range(self.images[0].size): self.W[i, i] = 0

    def recognize(self, corrupted, corr):
        prev = corrupted
        iter = 0
        while True:
            iter += 1
            diff = np.sum(np.abs(prev - corrupted))
            corrupted = self.W @ corrupted.reshape(corrupted.size, 1)
            # corrupted = np.where(corrupted > 0, 1, -1)
            corrupted = np.tanh(corrupted)
            corrupted = corrupted.flatten()
            # print(self.images[corr] - corrupted)
            # print((np.abs(self.images[corr] - corrupted) > 0.9).all())
            # to_return = self.m_in(corrupted)
            # if not to_return == False: return to_return, iter
            if self.mode == "projection":
                if (np.abs(np.abs(prev - corrupted)) < 0.001).all(): return self.m_in(corrupted), iter
            if self.mode == "easy":
                if (diff - np.sum(np.abs(prev - corrupted)) < 0.001): return None, iter
            prev = corrupted

    def m_in(self, image):
        if self.mode == "projection":
            image = np.where(image > 0, 1, -1)
            for i in range(self.images.shape[0]):
                if (self.images[i] == image).all():
                # if (np.abs(self.images[i] - image) > 0.9).all() and (np.abs(self.images[i] - image) < 1).all():
                    return i
            return False
        if self.mode == "easy":
            for i in range(self.images.shape[0]):
                if (self.images[i] == image).all():
                    return i
            return False

    def get_images(self):
        return self.images
