# -*- coding: utf-8 -*-
# @Time    : 2022/12/7 11:23
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : process.py
# @Software: PyCharm

import cv2


def img_process(img_url):
    img = cv2.imread(img_url)
    img = cv2.resize(img, (img.size))