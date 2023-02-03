import cv2
import time
import numpy as np
import onnxruntime
from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse
from .models import UploadIMG
from .form import ReviewForm

from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from rest_framework import permissions
from .serializers import UserSerializer, GroupSerializer

from .data_preprocess import *


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]


class GroupViewSet(viewsets.ModelViewSet):
    """
        API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]


running_time, model_name = 0, None
cloud_classification, cloud_segmentation_rate = None, None
de_cloud_model = onnxruntime.InferenceSession('./polls/MSBDN_RDFF_sim.onnx',
                                              providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

super_solution_model_ECBSR = onnxruntime.InferenceSession('./polls/ECBSR_x2_m16c64_prelu_rep.onnx',
                                                          providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
super_solution_model_EDSR = onnxruntime.InferenceSession('./polls/EDSR_sim.onnx',
                                                         providers=['CUDAExecutionProvider',
                                                                    'CPUExecutionProvider'])
super_solution_model_SwinIR = onnxruntime.InferenceSession('./polls/SwinIR_sim.onnx',
                                                           providers=['CUDAExecutionProvider',
                                                                      'CPUExecutionProvider'])

thin_cloud_classification_ConvNext = onnxruntime.InferenceSession('./polls/ConvNext_sim.onnx',
                                                                  providers=['CUDAExecutionProvider',
                                                                             'CPUExecutionProvider'])
cloud_segmentation_Unet = onnxruntime.InferenceSession('./polls/Segmentation_UNet_sim.onnx',
                                                       providers=['CUDAExecutionProvider',
                                                                  'CPUExecutionProvider'])


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


# def timm_data_transform():

# def ThinCloudClassification(request):


def SuperResolution(request):
    global running_time, model_name
    if request.method == 'POST':

        model_name = request.POST['model']

        assert model_name in ['ECBSR', 'EDSR', 'SwinIR']
        if model_name == 'ECBSR':
            super_solution_model = super_solution_model_ECBSR
        elif model_name == 'EDSR':
            super_solution_model = super_solution_model_EDSR
        elif model_name == 'SwinIR':
            super_solution_model = super_solution_model_SwinIR
        else:
            super_solution_model = None

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
        record_time_begin = time.time()
        high_resolution_img = super_solution_model.run(None, onnx_input)
        record_time_end = time.time()
        running_time = record_time_end - record_time_begin
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


def Cloud_Identification(request):

    global cloud_classification, cloud_segmentation_rate
    if request.method == 'POST':
        new_img = UploadIMG(
            img=request.FILES.get('img')
        )
        new_img.save()
        img_path = UploadIMG.objects.last().img.url
        img_path = r'O:/djangoProject' + img_path

        # 图像薄云判别
        img_timm_numpy, img_segmentation_numpy = data_preprocess_for_timm(img_path=img_path)

        onnx_input = {thin_cloud_classification_ConvNext.get_inputs()[0].name: img_timm_numpy}
        cloud_classification = thin_cloud_classification_ConvNext.run(None, onnx_input)
        cloud_classification = np.array(cloud_classification).reshape(2,)
        cloud_classification = np.exp(cloud_classification[0]) / np.exp(cloud_classification).sum()
        if cloud_classification >= 0.5:
            cloud_classification = '是'
        else:
            cloud_classification = '否'

        # 图像云层分割
        onnx_input = {cloud_segmentation_Unet.get_inputs()[0].name: img_segmentation_numpy}
        cloud_segmentation = cloud_segmentation_Unet.run(None, onnx_input)
        cloud_segmentation = (np.argmax(cloud_segmentation[0][0], axis=0) > 1).astype(np.uint8)

        cloud_segmentation_rate = cloud_segmentation.sum() / (256 ** 2) * 100

        # 修改名称

        cv2.imwrite(r'O:/djangoProject/' + 'polls/static/polls/raw_input.jpg', cv2.resize(cv2.imread(img_path), (512, 512)))
        cv2.imwrite(r'O:/djangoProject/' + 'polls/static/polls/segmentation_output.jpg',
                    cv2.resize(cloud_segmentation * 255, (512, 512)))

    return render(request, 'Cloud_Identification.html')


def showIdentificationResult(request):
    # 从数据的路径中载入低分图像

    return render(request, 'identification_result.html', context={'cloud_classification_result': cloud_classification,
                                                                  'cloud_segmentation_result': format(cloud_segmentation_rate, '.4f')})


def showResult(request):
    # 从数据的路径中载入低分图像

    return render(request, 'result.html', context={'running_model': model_name,
                                                   'running_time': format(running_time, '.4f')})


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
