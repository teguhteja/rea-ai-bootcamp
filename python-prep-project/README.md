# Preparation Python

## Preparation Python Project

### Description

To help you gain experience and gain knowledge in preparation for your journey in Mastering AI Bootcamp, we have prepared a simple project for you to get familiar with Python.

In this project, you will create a simple chatbot using the Gradio interface. We have prepared a skeleton code for you to work on in the file `main.py`. It contains the packages and some code that will help you focus on writing Python code.

First things first, run `pip install -r requirements.txt`. This will install all the necessary Python packages you'll need to complete the project.

For our chatbot, we will be using Huggingface model `microsoft/DialoGPT-medium`. We have also prepare the pipeline code where the chatbot is stored in the variable aptly named `chatbot`.

A list named `conversation_history` is provided to store the chat history. Please use this to store the chatbot history, both from the user and from the chatbot.

We have also provided the function for you to use, named `chatbot_response()` that receives one parameter `prompt`. Basically it receives a string that is stored in the variable `prompt`.

Underneath the function `chatbot_response()`, we have prepared a section for you to create your Gradio code. Look for `# Create a Gradio interface below` line.

Before we go into more details, We have also prepared a unit test that you can use, once you finished your code, you can validate it by running: `pytest -p no:warnings` from the command line. There are a total of 4 items that you must pass, so make sure you pass them all. If there are some tests that fails, read the error code and fix your code, then try again.

You have two tasks:
1. To create a Gradio interface for the chatbot interaction.
2. To create the interaction with the chatbot which we'll detail more below.

#### Chatbot Interaction
- User will input their interaction via a text box in Gradio which will then be passed to `chatbot_response()` function.
- The chatbot by default will pass the text into the chatbot that we have prepared using the variable named `chatbot`. You will then parse the response from the chatbot. Like this:
```python
outputs = chatbot(prompt, max_length=1000, pad_token_id=50256)
response = outputs[0]['generated_text'][len(prompt):].strip()
```
- You will then store the chatbot response into the list `conversation_history`.

The above is the normal interaction. There are 3 special cases that you need to build inside the `chatbot_response()` function:
1. If the input is `hello`, then the output must be `Hi there! How can I help you today?`, bypassing the chatbot.
2. If the input is `bye`, then the output must be `Goodbye! Have a great day!`, bypassing the chatbot.
3. If the input contains the word `calculate`, then the we must do the calculation using the following conditions:
    a. After the word `calculate`, it must be followed by a space then a number, a space, an operator, a space and a number. Example: `calculate 1 + 1`.
    b. The result of the calculation must be presented using the following text in front: `The result is` followed by a space then the result. Example: `The result is 2`.
    c. The calculation is limited to the operators `+`, `-`, `*`, `/`.
    d. If the operator is not one of the above and/or the format is not correct (not following point `a`), then the output must be: `Invalid operator and/or calculation format. Please use 'calculate <num1> <operator> <num2>'.`

Tips:
- Note that in coding, it must be exact, so if something is not right, check the text, maybe there is a typo, or it's missing `.` or `!` or `?`.


### Test Case Examples

#### Test Case 1

**Input**:

```txt
hello
what is AI?
bye
```

**Expected Output / Behavior**:

```txt
The conversation history should contain 6 entries: 3 prompts and 3 responses.
```

**Explanation**:

```txt
1. The first input "hello" should generate a response, adding two entries to the conversation history.
2. The second input "What is AI?" should generate a response, adding two more entries to the conversation history.
3. The third input "bye" should generate a response, adding the final two entries to the conversation history.
4. Therefore, the total length of the conversation history should be 6.
Because the response from the chatbot can be many things, in here we care about the conversation history only.
Naturally, we input the text one line by one line into the chatbot. Not all three directly in the input box inside Gradio.
```

#### Test Case 2

**Input**:

```txt
hello
```

**Expected Output / Behavior**:

```txt
"Hi there! How can I help you today?"
```

**Explanation**:

```txt
1. The input "hello" should trigger a predefined response "Hi there! How can I help you today?".
2. These responses are hardcoded and should always be the same for these inputs.
```

#### Test Case 3

**Input**:

```txt
bye
```

**Expected Output / Behavior**:

```txt
"Goodbye! Have a great day!"
```

**Explanation**:

```txt
1. The input "bye" should trigger a predefined response "Goodbye! Have a great day!".
2. These responses are hardcoded and should always be the same for these inputs.
```

#### Test Case 4

**Input**:

```txt
What is AI?
```

**Expected Output / Behavior**:

```txt
The response should not be empty. It can be anything from the chatbot model.
```

**Explanation**:

```txt
1. The input "What is AI?" should generate a response from the chatbot.
2. The response should contain some text, ensuring it is not empty.
```

#### Test Case 5

**Input**:

```txt
calculate 3 + 5
```

**Expected Output / Behavior**:

```txt
The result is 8
```

**Explanation**:

```txt
The input "calculate 3 + 5" should generate the response "The result is 8".
```

#### Test Case 6

**Input**:

```txt
calculate 5 - 3
```

**Expected Output / Behavior**:

```txt
The result is 2
```

**Explanation**:

```txt
The input "calculate 5 - 3" should generate the response "The result is 2".
```

#### Test Case 7

**Input**:

```txt
calculate 5 * 3
```

**Expected Output / Behavior**:

```txt
The result is 15
```

**Explanation**:

```txt
The input "calculate 5 * 3" should generate the response "The result is 15".
```

#### Test Case 8

**Input**:

```txt
calculate 16 / 2
```

**Expected Output / Behavior**:

```txt
The result is 8.0
```

**Explanation**:

```txt
The input "calculate 16 / 2" should generate the response "The result is 8.0".
```

#### Test Case 9

**Input**:

```txt
calculate 3 ^ 5
```

**Expected Output / Behavior**:

```txt
Invalid operator and/or calculation format. Please use 'calculate <num1> <operator> <num2>'.
```

**Explanation**:

```txt
The input "calculate 3 ^ 5" should generate an error response indicating an invalid operator.
```

#### Test Case 10

**Input**:

```txt
calculate 3 +
```

**Expected Output / Behavior**:

```txt
Invalid operator and/or calculation format. Please use 'calculate <num1> <operator> <num2>'.
```

**Explanation**:

```txt
The input "calculate 3 +" should generate an error response indicating an invalid calculation format.
```
