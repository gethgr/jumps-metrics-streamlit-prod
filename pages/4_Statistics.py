import time  # to simulate a real time data, time loop
import os
import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
import plotly.express as px
import plotly.graph_objects as go


############## ############## PAGE 4 STATISTICS ############# ############# ############## ########################
st.set_page_config(
    page_title="School of Physical Education and Sports Science Jumps Statistics",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Make the connection with Supabase - Database:
#@st.experimental_singleton
def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    #client = create_client(url, key)
    return create_client(url, key)
con = init_connection()

# @st.experimental_memo
def select_all_from_jumps_table():
    query=con.table("jumps_table").select("*").execute()
    return query
query = select_all_from_jumps_table()


df = pd.DataFrame(query.data)

st.markdown("<h1 style='text-align: left; color: black; font-weight:900'>Real-Time / Live Data Science Dashboard</h1>", unsafe_allow_html=True)
#st.title("Real-Time / Live Data Science Dashboard")

st.markdown("<h2 style='text-align: left; padding-top: 55px; color: #b4b406; font-weight:600'>Dashboard I</h1>", unsafe_allow_html=True)

# top-level filters
jump_trial_filter = st.selectbox("Select the type of the jump!", pd.unique(df["type_of_trial"]))

# creating a single-element container
placeholder = st.empty()

# dataframe filter
df_jumps_table = df[df["type_of_trial"] == jump_trial_filter]

# near real-time / live feed simulation
#for seconds in range(100):

#df_jumps_table["age_new"] = df_jumps_table["age"] * np.random.choice(range(1, 5))
#df_jumps_table["balance_new"] = df_jumps_table["balance"] * np.random.choice(range(1, 5))

# creating KPIs
total_trials = df_jumps_table['type_of_trial'].count()

avg_age = np.mean(df_jumps_table["age"])

avg_height = np.mean(df_jumps_table["height"])

avg_weight = np.mean(df_jumps_table["weight"])


# count_married = int(
#     df_jumps_table[(df_jumps_table["marital"] == "married")]["marital"].count()
#     + np.random.choice(range(1, 30))
# )

# balance = np.mean(df_jumps_table["balance_new"])

with placeholder.container():
    
    # create three columns
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    # fill in those three columns with respective metrics or KPIs
    kpi1.metric(
        label="Total Trials ✅",
        value= round(total_trials),
        delta=round(total_trials) - 1,

    )
    kpi2.metric(
        label="Average Age ✅",
        value=round(avg_age),
        delta=round(avg_age) - 10,
    )
    
    kpi3.metric(
        label="Average Height ✅",
        value=round(avg_height),
        delta=-10 + round(avg_height,3),
    )
    
    kpi4.metric(
        label="Average Weight ✅",
        value=int(avg_weight),
        delta=-10 + avg_weight,
    )
    
    st.markdown("""---""")

    # create two columns for charts
    fig_col1, fig_col2, fig_col3 = st.columns(3, gap='medium')
    with fig_col1:
        st.markdown("<h5 style='text-align: center; padding-top: 15px; color: Darkblue; font-weight:900'>Density Age-Height map.</h1>", unsafe_allow_html=True)
        #st.markdown("### Density Age-Height map!")
        fig = px.density_heatmap(
            data_frame=df_jumps_table, y="age", x="height"
        )
        fig.update_layout(
             margin=dict(l=0, r=0, t=10, b=60),
             #paper_bgcolor="LightSteelBlue",
        )
            
        st.plotly_chart(fig,use_container_width=True)     
           
    with fig_col2:
        st.markdown("<h5 style='text-align: center; padding-top: 15px; color: Darkblue; font-weight:900'>Density Age-Weight map.</h1>", unsafe_allow_html=True)
        #st.markdown("### Density Age-Weight map!")
        fig2 = px.density_heatmap(
            data_frame=df_jumps_table, y="age", x="weight"
        )
        fig2.update_layout(
             margin=dict(l=0, r=0, t=10, b=60),
             #paper_bgcolor="LightSteelBlue",
        )
        st.plotly_chart(fig2,use_container_width=True)

    with fig_col3:
        st.markdown("<h5 style='text-align: center; padding-top: 15px; color: Darkblue; font-weight:900'>Counts Per Age.</h1>", unsafe_allow_html=True)
        #st.markdown("### Counts Per Age!")
        fig3 = px.histogram(data_frame=df_jumps_table, x="age")
        fig3.update_layout(
             margin=dict(l=0, r=20, t=10, b=60),
             #paper_bgcolor="LightSteelBlue",
        )
        st.plotly_chart(fig3,use_container_width=True)
    
    

    st.markdown("<h2 style='text-align: left; padding-top: 55px; color: #b4b406; font-weight:600'>Dashboard II</h1>", unsafe_allow_html=True)

    st.markdown("""---""")


    col1, col2 = st.columns([1,2], gap="large")
    with col1:
        st.markdown("<h5 style='text-align: center; color: red; font-weight:900'>Total Count For Each Trial.</h1>", unsafe_allow_html=True)
        fig = px.pie(df, names='type_of_trial')
        fig.update_traces(hoverinfo='label+percent', textinfo='label+percent+value')
        fig.update_layout(
             #margin=dict(l=80, r=20, t=20, b=20),
             #paper_bgcolor="LightSteelBlue",
        )
        st.plotly_chart(fig,use_container_width=True)  
        
        
    with col2:
        st.markdown("<h5 style='text-align: left; color: red; font-weight:900'>Detailed All Data View.</h1>", unsafe_allow_html=True)
        st.dataframe(df[['id', 'fullname','occupy','type_of_trial','age','height','weight','email','created_at']])
        
          
    time.sleep(1)

    