# -*- coding: utf-8 -*-
# @Time    : 2022/11/22 16:21
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : urls.py
# @Software: PyCharm
from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
]