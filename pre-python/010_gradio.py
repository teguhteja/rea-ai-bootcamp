

# %% [markdown]
# # Gradio for simple user interface
# 
# Gradio is a Python library that allows you to quickly create customizable UI components around working with machine learning models. We already meet below code in the last learning:

# %%
# %pip install gradio

# %%
import gradio as gr
def greet(name):
    return "Hello " + name + "!"

gr.Interface(fn=greet, inputs="text", outputs="text").launch()

# %% [markdown]
# So now let's try to be more "machine learning" and use Gradio to create a simple user interface for a simple machine learning model.

# %% [markdown]
# # Challenge!
# 

# %% [markdown]
# From now on, you'll be challenged to:
# 
# - Try to solve problems with every concept you've learned so far
# - Learn to read and understand documentation and apply it to your code
# 
# For the first several case you'll be guided, but later on you'll be challenged to solve the problem on your own. Are you ready?!

# %% [markdown]
# ## Be a detective
# 
# When you're working with third-party libraries, you'll need several skills to make sure that you're doing things correctly:
# 
# - Understand the input that the library needs (the parameters)
# - Understand the output that the library gives (the return value)
# 
# Understand the input will help you to make sure your data is correclty formatted for the library to work with. Understand the output will help you to make sure you're using the library according to your requirements.
# 
# Identify below output of classifier

# %%
import gradio as gr
from transformers import pipeline

# Load the pipeline
classifier = pipeline('text-classification', model='SamLowe/roberta-base-go_emotions')

# Use the classifier with any text you want or you can try below suggestions
# - I am so happy
# - I'm really sorry to hear that

text = classifier("I am so happy")
print(text)

# %% [markdown]
# Identify the output first, then if you're ready, modify the code below so the output is formatted like so "label: score".
# 
# Note: `score` key is a float, not a string, so you should convert it to string first before concatenating it using `str()` function (e.g. `str(3.14)`)

# %%
def map_data_to_string_label(data):
    return f"{data[0]['label']}: {data[0]['score']}" # Your answer here

print(map_data_to_string_label(classifier("I'm really happy!"))) #The output should be "joy: 0.9066029787063599"

# %% [markdown]
# When you are done with the above challenge, then:
# 
# 1. Run the code block by pressing the play button.

# %%
# pip install rggrader

from rggrader import submit

# @title #### Student Identity
student_id = "REA6UCWBO" # @param {type:"string"}
name = "Ida Bagus Teguh Teja Murti" # @param {type:"string"}

# Submit Method
assignment_id = "012_gradio"
question_id = "01_map_data_to_string"

submit(student_id, name, assignment_id, str(map_data_to_string_label(classifier("I'm sad!"))), question_id)

# %% [markdown]
# After you've successfully modified the code, you should be able to run below Gradio interface without any error.

# %%
def classify_input(text):
    return map_data_to_string_label(classifier(text))

demo = gr.Interface(fn=classify_input, inputs="text", outputs="text").launch()

# %% [markdown]
# Cool right?!

# %% [markdown]
# # The stakeholder wants more! ⚠️⚠️⚠️⚠️
# 
# The stakeholder is happy with the result, but they want more! They want to be able to know other possible labels and their scores, not just the highest one.

# %%
print(classifier("I'm glad that you like it!", top_k=3))

# %% [markdown]
# So to be able to do that, you'll need to modify the code to use the `top_k` (What? Why `top_k`? What is that? Check notes on the bottom of this learning if you want to find out) parameter and we might set it to `3` so we can get the highest 3 labels and their scores.
# 
# But, if we add `top_k` parameter, you might notice that the output is now different, try it yourself!

# %% [markdown]
# ## Using `label`
# 
# ![image.png](attachment:image.png)
# 
# Because we want to output more than one score, we might consider to use a Gradio component that can display multiple values properly, like `label` component. As you can see above `label` component can be used to display multiple scores from a list of data, and the scores would be displayed as a bar chart that visually appealing.

# %% [markdown]
# ## Quick documentation runthrough
# 
# One skill that you'll need to master as machine learning engineer is to be able to read and understand documentation. For now, please take a look at this link: https://www.gradio.app/docs/label . As we've only a bit time left, let's quickly run through with the explanation:
# 
# `gr.Interface(fn=classify_input, inputs="text", outputs="text")`
# 
# First take a look on above code. We have two params that we want to highlight for now: `inputs` and `outputs`. This parameter should have a value of what's called in Gradio as "Component". In this case, we're using `text` component for both `inputs` and `outputs`.
# 
# The list of components can be found in the sidebar of the documentation page.
# 
# ![image.png](https://ik.imagekit.io/ffr6l4jaf5t/REA%20AI/image_UU_iaZNLM.png?updatedAt=1701664900534)
# 
# When you check at any component, the first thing you want to see is the "Shortcut" section. This section will tell you what to write if you want to use any component in `gr.Interface` function (Check the column "Interface String Shortcut", that string is what you need to write as either `inputs` or `outputs` parameter value).
# 
# Note: `text` is a shortcut for `textbox` component
# 
# ![image-2.png](https://ik.imagekit.io/ffr6l4jaf5t/REA%20AI/image_KH46tj99r.png?updatedAt=1701664418224)
# 
# Another thing that you want to check is the "Behavior" section. This section will help you to know the data type that the component will give as the `input` in `fn` function and what data type that we need to give as the `output`.
# 
# Behavior for `text`
# ![image-4.png](https://ik.imagekit.io/ffr6l4jaf5t/REA%20AI/image_G47j8qAuB.png?updatedAt=1701665053870)
# 
# Behavior for `label`
# ![image-3.png](https://ik.imagekit.io/ffr6l4jaf5t/REA%20AI/image_aGEiWrlxD.png?updatedAt=1701664711035)
# 
# If we have `text` as the `inputs` parameter, and `label` as the `outputs` parameter:
# 
# ```python
# def classify_input(text):
#     return {"positive": 0.9, "negative": 0.1}
# ```
# 
# The `text` parameter in `classify_input` function will be a string (referring to the documentation), and the return value should be a dictionary with the key will act as the label and the value will act as the score.

# %% [markdown]
# # Challenge!
# 
# Now that you already understand the `Label` component, let's try to make our Gradio interface to display the top 3 labels and their scores from previous classifier. First, make sure you understand the result of the `classifier` when we use `top_k` parameter.

# %%
print(classifier("I'm glad that you like it!", top_k=3))

# %% [markdown]
# Then, make sure that you can modify the output of the classifier to match the requirement of the label component (`Dict[str, float]`, basically a dictionary with string as the key and float as the value, eg: `{"excited": 0.9, "sad": 0.1}`).

# %%
def map_data_to_string_label(data):
    result = {}  # Initialize an empty dictionary
    for item in data:  # Iterate through the list of dictionaries
        result[item['label']] = item['score']  # Add label and score to the dictionary
    return result

print(map_data_to_string_label(classifier("I'm really happy!",top_k=3)))

# %% [markdown]
# When you are done with the above challenge, then:
# 
# 1. Run the code block by pressing the play button.

# %%
# Submit Method
assignment_id = "012_gradio"
question_id = "02_map_data_to_label"

submit(student_id, name, assignment_id, str(map_data_to_string_label(classifier("I'm sad!", top_k=3))), question_id)

# %% [markdown]
# After that combine everything you've learned to make a compatible data for `label` component as the return of `fn` function required by `gr.Interface` below. Good luck!

# %%
def classify_input(text):
    classification_result = classifier(text, top_k=3)  # Get predictions
    formatted_result = map_data_to_string_label(classification_result)  # Format the output
    return formatted_result  # Return the formatted result

# %% [markdown]
# When you are done with the above challenge, then:
# 
# 1. Run the code block by pressing the play button.

# %%
# Submit Method
assignment_id = "012_gradio"
question_id = "03_classify_input"

submit(student_id, name, assignment_id, str(classify_input("I'm sad!")), question_id)

# %%
#Run this code to enable Gradio
demo = gr.Interface(fn=classify_input, inputs="text", outputs="label").launch()


