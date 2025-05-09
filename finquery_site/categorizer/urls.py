from django.urls import path
from . import views

app_name = 'categorizer'

urlpatterns = [
    path('', views.categorizer_home, name='home'),
    path('bert/', views.categorizer_home, name='bert_home'),
    path('linear-svc/', views.linear_svc_home, name='linear_svc_home'),
    path('api/categorize/', views.api_categorize, name='api_categorize'),
    path('api/linear-svc/categorize/', views.api_linear_svc_categorize, name='api_linear_svc_categorize'),
]