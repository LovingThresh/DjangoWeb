# -*- coding: utf-8 -*-
# @Time    : 2022/11/22 16:21
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : urls.py
# @Software: PyCharm
from django.urls import path, re_path
from django.conf.urls.static import static
from django.conf import settings

from . import views

# register the app namespace
# URL NAMES

app_name = 'polls'

urlpatterns = [
    path('', views.index, name='index'),
    path('uploading', views.uploadImg, name='upload'),
    path('showing', views.showImg, name='show'),

    path('index_view', views.showIndex, name='index_view'),
    path('example', views.showExample, name='example'),


    path('SuperResolution', views.SuperResolution, name='SuperResolution'),

    path('<int:question_id>/', views.detail, name='detail'),
    path('<int:question_id>/results/', views.results, name='results'),
    path('<int:question_id>/vote/', views.vote, name='vote')
]
