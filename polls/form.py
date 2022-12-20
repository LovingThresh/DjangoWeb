# -*- coding: utf-8 -*-
# @Time    : 2022/12/20 21:04
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : form.py
# @Software: PyCharm
from django import forms


class ReviewForm(forms.Form):
    first_name = forms.CharField(label='First Name', max_length=100)
    last_name = forms.CharField(label='Last Name', max_length=100)
    email = forms.EmailField(label='email')
    review = forms.CharField(label='Please write your review here')
