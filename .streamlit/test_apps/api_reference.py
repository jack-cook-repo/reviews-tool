import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time

'''
# Magic commands

This is some _markdown_.
'''

df = pd.DataFrame({'col1': [1,2,3]})
df  # <-- Draw the dataframe

x = 10
'x has a value of ', x  # <-- Draw the string 'x' and then the value of x

'''
# Display text

## Header

### Sub header
'''

code = '''def hello():
    print("Hello, Streamlit!")'''
st.write('We can use st.code() to write out code blocks')
st.code(code, language='python')

'''
# Display charts
'''
arr = np.random.normal(1, 1, size=100)
# Using matplotlib
fig, ax = plt.subplots()
sns.histplot(arr, bins=20, ax=ax)
st.pyplot(fig)


'''
# Display media
'''
image_url = 'https://www.lambocars.com/wp-content/uploads/2021/04/1982_countach_1_of_3_1.jpg'
st.image(image_url, caption='A lambo')

video_url = 'https://www.youtube.com/watch?v=ryXNmprejvo'
st.video(video_url, start_time=1000)


'''
# Display widgets

## First, demo callbacks
1. Set up session_state for a variable
2. Create a function that alters that session_state variable
3. Instantiate a widget, calling the function in on_click
4. Add logic for 'if widget is used, then write output'
'''
if 'callback_str' not in st.session_state:
    st.session_state.callback_str = ''


def callback_function(num, text):
    st.session_state.callback_str = f'Callback function used with num {str(num)} and text {text}'


# In this example, we use args and kwargs separately - in reality we'd probably just use one or the other.
my_button = st.button(label='A button label',
                      key='my_button',  # Used in st.session_state
                      help='A tooltip you could use',
                      on_click=callback_function,
                      args=(3, ),  # Used as arg for num
                      kwargs={'text': 'Test text'})  # Used as kwarg for 'text'

if my_button:
    st.write(st.session_state.callback_str)


'## Checkbox\n'
agree = st.checkbox('I agree')
if agree:
    st.write('Great!')