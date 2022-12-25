from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('',views.home,name='home'),
    path('hashtag_analysis',views.hashtag_analysis,name='hashtag_analysis'),
    path('tweet_analysis',views.tweet_analysis,name='tweet_analysis'),
    path('analysis',views.analysis,name='analysis'),
    path('sentiment',views.sentiment,name='sentiment'),
]