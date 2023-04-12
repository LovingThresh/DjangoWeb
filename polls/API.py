# -*- coding: utf-8 -*-
# @Time    : 2023/4/10 11:41
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : API.py
# @Software: PyCharm
# 实现API接口，接收base64，输出输入的base64

import base64
import datetime

from .models import ImageAPIModel

from rest_framework.response import Response
from rest_framework import serializers, viewsets
from django.core.files.base import ContentFile


class ImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ImageAPIModel
        fields = ('id', 'name', 'image')


class ImageViewSet(viewsets.ModelViewSet):
    queryset = ImageAPIModel.objects.all()
    serializer_class = ImageSerializer

    def create(self, request, *args, **kwargs):
        image_data = base64.b64decode(request.data['image'])
        image_name = request.data['name']
        photo = ImageAPIModel.objects.create(name=image_name)
        photo.image.save(image_name, ContentFile(image_data), save=True)
        serializer = ImageSerializer(photo)
        return Response(serializer.data)


# with open('image.png', 'rb') as f:
#     image_data = f.read()
#     base64_string = base64.b64encode(image_data).decode('utf-8')
#
# data = {'image': base64_string}
