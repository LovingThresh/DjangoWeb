"""djangoProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include, re_path
from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings
from django.conf.urls.static import static


def home_view(request):
    var = {'hello': 'Liuye'}
    return render(request, 'runoob.html', context=var)


def simple_view(request):
    return render(request, 'result.html')


def login_view(request):
    return render(request, 'logining.html')


urlpatterns = [
                  path("", home_view, name='home'),
                  path("login/", login_view, name='login'),
                  path("result/", simple_view, name='result'),
                  path("admin/", admin.site.urls),
                  path('polls/', include('polls.urls')),
              ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
