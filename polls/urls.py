# -*- coding: utf-8 -*-
# @Time    : 2022/11/22 16:21
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : urls.py
# @Software: PyCharm
from django.urls import path

from . import views

# register the app namespace
# URL NAMES

app_name = 'polls'

urlpatterns = [
    path('', views.showIndex, name='index'),
    # path('logining', views.show_logining),
    path('login', views.login, name='login'),
    path('uploading', views.uploadImg, name='upload'),
    path('showing', views.showImg, name='show'),
    path('example', views.showExample, name='example'),

    path('SuperResolution', views.SuperResolution, name='SuperResolution'),
    path('Decloud', views.DeCloud, name='Decloud'),
    path('Cloud_Identification', views.Cloud_Identification, name='Cloud_Identification'),

    path('result', views.showResult, name='result'),
    path('identification_result', views.showIdentificationResult, name='identification_result'),

    path('thank_you', views.thank_you, name='thank_you')
]
