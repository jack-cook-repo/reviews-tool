import streamlit as st
import numpy as np
import pandas as pd

st.write('# Uber pickups in NYC')

# Load Uber dataset
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')


@st.cache  # This allows us to cache results so that we don't have to keep re-running it
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    # Turn columns into lowercase names
    data = data.rename(str.lower,
                       axis='columns')
    data['date/time'] = pd.to_datetime(data['date/time'])
    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Load 10,000 rows of data into the dataframe.
data = load_data(nrows=10000)

# Notify the reader that the data was successfully loaded, overwriting old text
data_load_state.write('Loading data...done!')

# Now let's write some data
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

# And then plot a chart of pickup times
st.subheader('Number of pickups by hour')
hist_values = np.histogram(data['date/time'].dt.hour,
                           bins=24,
                           range=(0, 24))[0]
st.bar_chart(hist_values)

# Then, look at locations for different times of day
hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
filtered_data = data[data['date/time'].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)

