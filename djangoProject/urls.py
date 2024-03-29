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
from django.urls import path, include
from django.shortcuts import render
from django.conf import settings
from django.conf.urls.static import static

from rest_framework import routers

import polls.views
from polls import views, API

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'groups', views.GroupViewSet)
router.register(r'image', API.ImageViewSet)


def simple_view(request):
    return render(request, 'result.html')


urlpatterns = [
                  path("", polls.views.login, name='login'),
                  path("result/", simple_view, name='result'),
                  path("admin/", admin.site.urls),
                  path('polls/', include('polls.urls')),

                  path('', include(router.urls)),
                  path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))

              ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
