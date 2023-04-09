# -*- coding: utf-8 -*-
# @Time    : 2023/4/9 17:45
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : sign_up.py
# @Software: PyCharm
from django.contrib.auth.models import User
user = User.objects.create_user("liuye@163.com", "liuye@163.com", "13104898753")
user.save()
