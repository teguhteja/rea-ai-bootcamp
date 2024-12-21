# test_main.py
import pytest
from main import chatbot_response, conversation_history

def test_conversation_history():
    chatbot_response("hello")
    chatbot_response("What is AI?")
    chatbot_response("bye")
    assert len(conversation_history) == 6  # 3 prompts + 3 responses

def test_predefined_responses():
    response = chatbot_response("hello")
    assert "Hi there! How can I help you today?" in response
    
    response = chatbot_response("bye")
    assert "Goodbye! Have a great day!" in response

def test_model_response():
    response = chatbot_response("What is AI?")
    assert len(response) > 0  # Check if response is not empty

def test_calculate_responses():
    chatbot_response("calculate 3 + 5")
    assert "The result is 8" in conversation_history[-1]
    
    chatbot_response("calculate 10 - 2")
    assert "The result is 8" in conversation_history[-1]
    
    chatbot_response("calculate 4 * 2")
    assert "The result is 8" in conversation_history[-1]
    
    chatbot_response("calculate 16 / 2")
    assert "The result is 8.0" in conversation_history[-1]
    
    chatbot_response("calculate 3 ^ 5")
    assert "Invalid operator and/or calculation format. Please use 'calculate <num1> <operator> <num2>'." in conversation_history[-1]
    
    chatbot_response("calculate 3 +")
    assert "Invalid operator and/or calculation format. Please use 'calculate <num1> <operator> <num2>'." in conversation_history[-1]
