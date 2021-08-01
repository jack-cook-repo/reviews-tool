import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

# Perform query.
# Uses st.cache to only rerun when the query changes or after 10 min (600 seconds).
@st.cache(ttl=600)
def run_query(query):
    query_job = client.query(query)
    rows_raw = query_job.result()
    # Convert to list of dicts. Required for st.cache to hash the return value.
    rows = [dict(row) for row in rows_raw]
    return rows


rows = run_query("SELECT * FROM `crested-unity-321617.analytics.chicaco_crime_sample` LIMIT 10")

# Convert from list of dicts to dataframe
df_rows = pd.DataFrame(rows)

st.write(f'BigQuery results:')
# Print results.
st.write(df_rows)