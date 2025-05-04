 
from flask import Blueprint, request, jsonify
from .chatbot import get_chat_response

main = Blueprint('main', __name__)

@main.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query")
    response = get_chat_response(user_query)
    return jsonify({"response": response})
