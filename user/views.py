from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from .serializer import RegisterSerializer
from rest_framework.permissions import IsAuthenticated
from .llm import ask_llm
from .models import ChatMessage



class RegisterView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)

        if serializer.is_valid():
            serializer.save()
            return Response(
                {"message": "User created successfully"},
                status=status.HTTP_201_CREATED
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
class ProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response({
            "username": request.user.username,
            "email": request.user.email
        })
class ChatView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user
        message_text = request.data.get("message")
        ChatMessage.objects.create(
            user=user,
            role="user",
            content=message_text
        )
        history = ChatMessage.objects.filter(user=user).order_by("-created_at")[:10]
        history = list(reversed(history))

        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in history
        ]
        reply = ask_llm(messages)
        ChatMessage.objects.create(
            user=user,
            role="assistant",
            content=reply
        )

        return Response({"reply": reply})
class ChatHistoryView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        messages = ChatMessage.objects.filter(user=request.user).order_by("created_at")

        return Response([
            {"role": m.role, "text": m.content}
            for m in messages
        ])