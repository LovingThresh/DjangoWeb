import os
import cv2
import numpy as np
import onnxruntime
from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse
from .models import UploadIMG
from .form import ReviewForm

de_cloud_model = onnxruntime.InferenceSession('./polls/MSBDN_RDFF_sim.onnx',
                                              providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
super_solution_model = onnxruntime.InferenceSession('./polls/ECBSR_x2_m16c64_prelu_rep.onnx',
                                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])


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

    return render(request, 'result.html')


def SuperResolution(request):
    if request.method == 'POST':
        new_img = UploadIMG(
            img=request.FILES.get('img')
        )
        new_img.save()
        img_path = UploadIMG.objects.last().img.url
        img_path = r'O:/djangoProject' + img_path

        # 图像超分
        low_resolution_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        show_low_resolution_img = cv2.resize(low_resolution_img, (512, 512))

        low_resolution_img = cv2.cvtColor(low_resolution_img, cv2.COLOR_BGR2RGB)
        low_resolution_img = np.expand_dims((low_resolution_img / 255).transpose(2, 0, 1), 0).astype(np.float32)

        onnx_input = {super_solution_model.get_inputs()[0].name: low_resolution_img}
        high_resolution_img = super_solution_model.run(None, onnx_input)
        high_resolution_img = np.uint8(np.clip(high_resolution_img[0].squeeze(0).transpose(1, 2, 0), 0, 1) * 255)
        high_resolution_img = cv2.cvtColor(high_resolution_img, cv2.COLOR_RGB2BGR)
        show_high_resolution_img = cv2.resize(high_resolution_img, (512, 512))

        # 修改名称
        cv2.imwrite(r'O:/djangoProject/' + 'polls/static/polls/raw_result.jpg', show_low_resolution_img)
        cv2.imwrite(r'O:/djangoProject/' + 'polls/static/polls/generated_result.jpg', show_high_resolution_img)

    return render(request, 'SuperResolution.html')


def DeCloud(request):
    if request.method == 'POST':
        new_img = UploadIMG(
            img=request.FILES.get('img')
        )
        new_img.save()
        img_path = UploadIMG.objects.last().img.url
        img_path = r'O:/djangoProject' + img_path

        # 图像超分
        cloud_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        cloud_img = cv2.resize(cloud_img, (256, 256))
        show_cloud_img = cv2.resize(cloud_img, (512, 512))

        cloud_img = cv2.cvtColor(cloud_img, cv2.COLOR_BGR2RGB)
        cloud_img = np.expand_dims(((cloud_img - 127.5) / 127.5).transpose(2, 0, 1), 0).astype(np.float32)

        onnx_input = {de_cloud_model.get_inputs()[0].name: cloud_img}
        de_cloud_img = de_cloud_model.run(None, onnx_input)
        de_cloud_img = np.uint8(np.clip(de_cloud_img[0].squeeze(0).transpose(1, 2, 0), -1, 1) * 127.5 + 127.5)
        de_cloud_img = cv2.cvtColor(de_cloud_img, cv2.COLOR_RGB2BGR)
        show_de_cloud_img = cv2.resize(de_cloud_img, (512, 512))

        # 修改名称
        cv2.imwrite(r'O:/djangoProject/' + 'polls/static/polls/raw_result.jpg', show_cloud_img)
        cv2.imwrite(r'O:/djangoProject/' + 'polls/static/polls/generated_result.jpg', show_de_cloud_img)

    return render(request, 'Decloud.html')


def form(request):
    # POST  REQUEST --> FORM CONTENT
    if request.method == 'POST':
        form = ReviewForm(request.POST)
        if form.is_valid():
            print(form.cleaned_data)
            return redirect(reverse('polls:thank_you'))
    else:
        form = ReviewForm()
    return render(request, 'form.html', context={'form': form})


def thank_you(request):
    return render(request, 'thank_you.html')


def show_logining(request):
    return render(request, 'logining.html')

# def Show_SuperResolution(request):
#     last_image_path = UploadIMG.objects.last().img.url
#     processed_img = img_process()
