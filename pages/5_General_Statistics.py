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
st.markdown("# Real-Time / Live Data Science Dashboard")

st.markdown("### Dashboard I")

# top-level filters
columns = [" "] + [df.columns[i] for i in range (0, len(df.columns))]
options = (" ", "CMJ", "DJ", "ISO", "SJ")
jump_trial_filter = st.selectbox("Select the type of the trial:", options)

# creating a single-element container
placeholder = st.empty()

# dataframe filter
if jump_trial_filter != " ":
    df_jumps_table = df[df["type_of_trial"] == jump_trial_filter]
    total_trials = df_jumps_table['type_of_trial'].count()
    avg_age = np.mean(df_jumps_table["age"])
    avg_height = (np.mean(df_jumps_table["height"]))
    avg_weight = round(np.mean(df_jumps_table["weight"]),3)

    with placeholder.container():
        # create three columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            label="Total Trials ✅",
            value= round(total_trials),
            delta=round(total_trials) - 1,
        )
        col2.metric(
            label="Average Age ✅",
            value=round(avg_age),
            delta=round(avg_age) - 10,
        )
        col3.metric(
            label="Average Height ✅",
            value=round(avg_height),
            delta=-10 + round(avg_height,3),
        )
        col4.metric(
            label="Average Weight ✅",
            value=int(avg_weight),
            delta=-10 + avg_weight,
        )
        st.markdown("""---""")

        # create two columns for charts
        fig_col1, fig_col2, fig_col3 = st.columns(3, gap='medium')
        with fig_col1:
            st.markdown("##### Density Age-Height map")
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
            st.markdown("##### Density Age-Weight map")
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
            st.markdown("##### Counts Per Age")
            #st.markdown("### Counts Per Age!")
            fig3 = px.histogram(data_frame=df_jumps_table, x="age")
            fig3.update_layout(
                margin=dict(l=0, r=20, t=10, b=60),
                #paper_bgcolor="LightSteelBlue",
            )
            st.plotly_chart(fig3, use_container_width=True)

st.write("### Dashboard II")
#st.markdown("---")

col1, col2 = st.columns([1,2], gap="large")
with col1:
    st.markdown("Total count for each trial:")
    fig = px.pie(df, names='type_of_trial')
    fig.update_traces(hoverinfo='label+percent', textinfo='label+percent+value')
    fig.update_layout(
        #margin=dict(l=80, r=20, t=20, b=20),
        #paper_bgcolor="LightSteelBlue",
    )
    st.plotly_chart(fig,use_container_width=True)  
    
with col2:
    df.rename(columns = {'id' :'ID',
                     'created_at':'Created at',
                     'fullname':'Fullname', 
                     'email':'Email',
                     'occupy':'Occupy',
                     'type_of_trial':'Type of trial',
                     'filename':'Filename',
                     'filepath':'Filepath',
                     'height':'Height',
                     'weight':'Weight',
                     'age':'Age',
                     'instructor':'Instructor',
                     'drop_height':'Drop height'

    }, inplace = True)
    st.markdown("Detailed all data view:")
    st.dataframe(df[['ID', 'Fullname','Occupy','Type of trial','Age','Height','Weight','Email','Created at']].sort_values('Created at', ascending=False), use_container_width = True)
    
time.sleep(1)

    