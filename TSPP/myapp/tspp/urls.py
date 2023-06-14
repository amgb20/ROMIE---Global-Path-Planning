from django.urls import path
from . import views
from .views import mapview

urlpatterns = [
    path('home/', views.home, name='home'),
    path('index/', views.index, name='index'),
    path('tspp_results/', views.tspp_results, name='tspp_results'),
    path('', views.landingpage, name='landingpage'),
    path('map/', mapview, name='map'),
    path('solve-tsp/', views.solve_tsp, name='solve-tsp'),
    path('download_path_csv/<str:algorithm>/<int:Length>/<int:Width>/', views.download_path_csv, name='download_path_csv'),
    path('download_csv/<str:algorithm>/<int:Length>/<int:Width>/', views.download_elapsed_time_csv, name='download_csv'),
    path('download_cpu_csv/<str:algorithm>/<int:Length>/<int:Width>/', views.download_cpu_usages_csv, name='download_cpu_csv'),
]

