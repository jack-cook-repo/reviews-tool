import streamlit as st
import numpy as np
import pandas as pd

st.write('# This is a title written in markdown')

'''
## This is a sub title written with magic commands.

*This* was _also_ written by magic commands.
'''

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

# Can either use st.write(df), or simply just call the value
df

option = st.sidebar.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option

if st.sidebar.checkbox('Click to show chart'):
    chart_data = pd.DataFrame(
         np.random.randn(20, 3),
         columns=['a', 'b', 'c'])

    st.line_chart(chart_data)
else:
    st.write('\nUse the checkbox in the side bar to show the chart\n')

# We can also create maps
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)