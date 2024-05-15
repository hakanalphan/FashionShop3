from django.urls import path
from . import views


from MainPage import views
urlpatterns = [
    path('', views.index, name='index'),

]