# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 16:12
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : request_API.py
# @Software: PyCharm
import base64
import datetime
import requests


with open('O:/djangoProject/API.png', 'rb') as f:
    image_data = f.read()
    base64_string = base64.b64encode(image_data).decode('utf-8')

data = {'name': datetime.datetime.now().strftime('%Y%m%d%H%M%S%f') + '.png', 'image': base64_string}
url = 'http://127.0.0.1:8000/api/images/'

response = requests.post(url, data=data)

print(response.json())
