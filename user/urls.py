from django.urls import path
from .views import RegisterView, ProfileView,ChatView,ChatHistoryView

urlpatterns = [
    path("register/", RegisterView.as_view(), name="register"),
    path("profile/", ProfileView.as_view()),
    path("chat/", ChatView.as_view()),
    path("chat/history/", ChatHistoryView.as_view()),

]