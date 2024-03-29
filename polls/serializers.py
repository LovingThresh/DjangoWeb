# -*- coding: utf-8 -*-
# @Time    : 2022/12/31 10:20
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : serializers.py
# @Software: PyCharm
from django.contrib.auth.models import User, Group
from rest_framework import serializers


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ['url', 'username', 'email', 'groups']


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ['url', 'name']


class ImageSerializer(serializers.Serializer):
    # 定义一个图像字段
    image = serializers.ImageField()
#     image_base = serializers.JSONBoundField()

    # 定义一个结果字段
    result = serializers.CharField()
