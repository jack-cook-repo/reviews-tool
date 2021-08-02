import streamlit as st
import numpy as np
import time

'''# Batch elements and input widgets

Using st.form you can batch input widgets together and along with st.form_submit_button submit the state 
inside these widgets with the click of a single button.
'''
# Forms can be declared using the 'with' syntax
with st.form(key='my_form'):
    text_input = st.text_input(key='name', label='Enter your name')
    submit_button = st.form_submit_button(label='Submit')

# st.form_submit_button returns True upon form submit
if submit_button:
    st.write(f'Hello {st.session_state.name}')

'Alternative syntax, declare a form and use the returned object'
form = st.form(key='my_other_form')
form.text_input(label='Enter some text')
submit_button_new = form.form_submit_button(label='Submit')


'# Insert elements out of order'

# Appends some text to the app.
st.text('This will appear first')

# Appends an empty slot to the app. We'll use this later.
my_slot1 = st.empty()

# Appends another empty slot.
my_slot2 = st.empty()

# Appends some more text to the app.
st.text('This will appear last')

# Replaces the first empty slot with a text string.
my_slot1.text('This will appear second')

# Replaces the second empty slot with a chart.
my_slot2.line_chart(np.random.randn(20, 2))


'# Animate elements'
progress_bar = st.progress(0)
status_text = st.empty()
chart = st.line_chart(np.random.randn(10, 2))

for i in range(20):
    # Update progress bar.
    progress_bar.progress(5*i + 5)

    new_rows = np.random.randn(10, 2)

    # Update status text.
    status_text.text(
        'The latest random number is: %s' % new_rows[-1, 1])

    # Append data to the chart.
    chart.add_rows(new_rows)

    # Pretend we're doing some computation that takes time.
    time.sleep(0.01)

status_text.text('Done!')


'# Append data to a table or a chart'
# Get some data.
data = np.random.randn(10, 2)

# Show the data as a chart.
chart = st.line_chart(data)

# Wait 1 second, so the change is clearer.
time.sleep(1)

# Grab some more data.
data2 = np.random.randn(10, 2)

# Append the new data to the existing chart.
chart.add_rows(data2)
