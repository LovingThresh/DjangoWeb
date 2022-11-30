from django.shortcuts import render
from django.http import HttpResponse
from .models import UploadIMG


# Create your views here.
def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def detail(request, question_id):
    return HttpResponse("You're looking at question %s." % question_id)


def results(request, question_id):
    response = "You're looking at the results of question %s."
    return HttpResponse(response % question_id)


def vote(request, question_id):
    return HttpResponse("You're voting on question %s." % question_id)


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
