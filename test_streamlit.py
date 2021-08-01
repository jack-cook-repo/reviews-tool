import streamlit as st

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

# Lay out columns side by side
left_column, right_column = st.beta_columns(2)
pressed = left_column.button('Press me?')
if pressed:
    right_column.write("Woohoo!")

# Expanders can hide page content
expander = st.beta_expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")

# Remember, we could use .write() here, but just calling the string also workds
'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.02)

# Print after we're done
'...and now we\'re done!'