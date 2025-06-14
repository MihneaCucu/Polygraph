from django.urls import path
from .views import index, analysis_view, feedback_view, feedback_success_view, result_view, get_summary, entry_list

# Lista cu url-uri din pagina_principala app
urlpatterns = [
    path('', index, name="index"),
    path('analyze/', analysis_view, name="analyze"),
    path('feedback/', feedback_view, name="feedback"),
    path('feedback_success/', feedback_success_view, name="feedback-success"),
    path('news_result/', result_view, name="result"),
    path('get-summary/', get_summary, name='get_summary'),
    path('entries/', entry_list, name='entry-list'),
]

