import streamlit as st
import numpy as np
import pandas as pd

'''
# Examples of displaying
'''
# Generate 10 x 20 random number dataframe sampled from standard normal distribution
np.random.seed(47)
dataframe = pd.DataFrame(np.random.randn(10, 20),
                         columns=('col %d' % i for i in range(20)))

# highlight_max allows you to take an axis, identify the largest value, and highlight the result
st.dataframe(dataframe.style.highlight_max(axis=0, color='blue'))

# Static table styling
st.table(dataframe.loc[:3, :'col 3'])


'''
# Examples of widgets
'''
x = st.slider('x', key="test_slider")  # 👈 this is a widget
st.write(x, 'squared is', x**2)

# We can also access the inputs/outputs of a widget using the key
st.write('Value of slider input is: ', st.session_state.test_slider)

'''
# Examples of layout

## Sidebar testing
'''
if st.checkbox('Click to show trial sidebar elements'):
    # Add a selectbox to the sidebar:
    add_selectbox = st.sidebar.selectbox(
        'How would you like to be contacted?',
        ('Email', 'Home phone', 'Mobile phone')
    )

    # Add a slider to the sidebar:
    add_slider = st.sidebar.slider(
        'Select a range of values',
        0.0,  # Min
        100.0,  # Max
        (25.0, 75.0)  # Default - in this case, 2 values!
    )

'## Side by side columns'
left_column, right_column = st.beta_columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")