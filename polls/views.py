import os

from django.shortcuts import render
from django.http import HttpResponse
from .models import UploadIMG

import cv2


# Create your views here.

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


# Create your views here.
def uploadImg(request):
    if request.method == 'POST':
        new_img = UploadIMG(
            img=request.FILES.get('img')
        )
        new_img.save()
    return render(request, 'uploading.html')


def showImg(request):
    imgs = UploadIMG.objects.all()
    content = {
        'imgs': imgs.first().img.url,
    }
    return render(request, 'showing.html', content)


def showIndex(request):
    return render(request, 'index.html')


def showExample(request):
    return render(request, 'example.html')


def showResult(request):
    # 从数据的路径中载入低分图像
    img_path = UploadIMG.objects.last().img.url
    print(img_path)
    img_path = r'O:/djangoProject' + img_path
    # 图像处理
    print(img_path)
    high_resolution_img = cv2.imread(img_path)
    high_resolution_img = cv2.resize(high_resolution_img, (512, 512))

    low_resolution_img = cv2.resize(high_resolution_img, (256, 256))
    low_resolution_img = cv2.resize(low_resolution_img, (512, 512))

    # 修改名称
    cv2.imwrite(r'O:/djangoProject/' + 'polls/static/polls/low_resolution.jpg', low_resolution_img)
    cv2.imwrite(r'O:/djangoProject/' + 'polls/static/polls/high_resolution.jpg', high_resolution_img)

    return render(request, 'result.html')


def SuperResolution(request):
    if request.method == 'POST':
        new_img = UploadIMG(
            img=request.FILES.get('img')
        )
        new_img.save()

    return render(request, 'SuperResolution.html')

# def Show_SuperResolution(request):
#     last_image_path = UploadIMG.objects.last().img.url
#     processed_img = img_process()
