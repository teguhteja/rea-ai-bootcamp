how to run it in google collab
why use google collab ? for comp dont have gpu. you can use it 

how to use it ?
1. create google collab
2. upload requirements.txt, test_main.py, main.py
3. run !pip install -r requirements.txt
4. run pytest test_main.py to looking error
5. edit main.py based https://huggingface.co/microsoft/DialoGPT-medium from edit menu
6. final edit 
```py
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import re

# Load the Hugging Face model and tokenizer
device = 0 if torch.cuda.is_available() else -1
model_name = "microsoft/DialoGPT-medium"
chatbot = pipeline("text-generation", model=model_name, device=device)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize conversation history
conversation_history = []
chat_history_ids = None

# Function to generate chatbot response
def chatbot_response(prompt):
    global conversation_history, chat_history_ids

    #write your code below
    if prompt.lower() == "hello":
        response = "Hi there! How can I help you today?"
    elif prompt.lower() == "bye":
        response = "Goodbye! Have a great day!"
    elif "calculate" in prompt.lower():
        match = re.match(r"calculate (\d+) ([+\-*/]) (\d+)", prompt.lower())
        if match:
            num1, operator, num2 = match.groups()
            num1, num2 = int(num1), int(num2)
            if operator == '+':
                result = num1 + num2
            elif operator == '-':
                result = num1 - num2
            elif operator == '*':
                result = num1 * num2
            elif operator == '/':
                result = num1 / num2
            response = f"The result is {result}"
        else:
            response = "Invalid operator and/or calculation format. Please use 'calculate <num1> <operator> <num2>'."
    else:
        # Use the chatbot model for other inputs
        # outputs = chatbot(prompt, max_length=1000, pad_token_id=50256)
        # response = outputs[0]['generated_text'][len(prompt):].strip()
        new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)  
        else : 
            bot_input_ids = new_user_input_ids
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
        response =tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True) 
        print("DialoGPT: {}".format(response))

    # Update conversation history
    conversation_history.append(f"User: {prompt}")
    conversation_history.append(f"Bot: {response}")


    # Display conversation history (loops and data structures)
    history = "\n".join(conversation_history[-6:])  # Show last 3 interactions
    
    return history

# Create a Gradio interface below
if __name__ == "__main__":

  iface = gr.Interface(  # Make sure this line is indented correctly
        fn=chatbot_response,
        inputs=gr.Textbox(lines=2, placeholder="Enter your message here..."),
        outputs="text",
        title="Chatbot Interface",
        description="A simple chatbot interface using Gradio and Hugging Face's DialoGPT."
    )
  iface.launch(share=True) 
```
7. run main.py and insert some text

trade-off :
walaupun menggunakan google collab namun ada yang ditukarkan yaitu file yang kita edit atau rubah harus disimpan di local karena setiap 3 jam akan dihapus. anda dapat menyimpan file anda bila menggunakan google collab yang dibayarkan