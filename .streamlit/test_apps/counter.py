import streamlit as st
import datetime

st.write('# Counter Example')

# Check if 'count' already exists in session_state
# If not, then initialize it as 0, and initialise late_updated
if 'count' not in st.session_state:
    # Could also use st.session_state.count = 'value'
    st.session_state['count'] = 0
    st.session_state['count_new'] = 0
    st.session_state['last_updated'] = datetime.time(0, 0)


'## Example 1: callbacks'
def alter_counter(num=0, reset_value=False):
    if reset_value:
        st.session_state.count = 0
    else:
        st.session_state.count += num


# Get amount to increment by
increment_value = st.number_input('Enter a value', value=0, step=1)

left_column, right_column = st.beta_columns(2)
# Callbacks allow you to define a function to run when a widget is interacted with
increment = left_column.button('Increment',
                               on_click=alter_counter,
                               args=(increment_value, ))  # Must be iterable, hence forced tuple
reset = right_column.button('Reset count',
                            on_click=alter_counter,
                            kwargs=({'reset_value': True}))

st.write('Count = ', st.session_state['count'])


'## Example 2: callbacks and forms'
def update_counter():
    st.session_state.count_new += st.session_state.increment_value
    st.session_state.last_updated = st.session_state.update_time


with st.form(key='my_form'):
    st.time_input(label='Enter the time', value=datetime.datetime.now().time(), key='update_time')
    st.number_input('Enter a value', value=0, step=1, key='increment_value')
    submit = st.form_submit_button(label='Update', on_click=update_counter)

st.write('Current Count = ', st.session_state.count_new)
st.write('Last Updated = ', st.session_state.last_updated)


'## Example 3: session state and widget state association'
if "celsius" not in st.session_state:
    # set the initial default value of the slider widget
    st.session_state.celsius = 50.0

st.slider(
    "Temperature in Celsius",
    min_value=-100.0,
    max_value=100.0,
    key="celsius"
)

# This will get the value of the slider widget
st.write(st.session_state.celsius)