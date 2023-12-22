import os
import streamlit as st
import pandas as pd
from supabase import create_client, Client
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from typing import List, Tuple


# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestRegressor

# calculate the Pearson's correlation between two variables
from numpy.random import randn
from numpy.random import seed
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from numpy import cov




       
###############-------SET PAGE CONFIGURATION--------##########
st.set_page_config(
    page_title="Trials Metrics Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

###############-------END OF SET PAGE CONFIGURATION--------##########


#############-------CONNECTION WITH DATABASE/SUPABASE--------############
#Make the connection with Supabase - Database:
#@st.experimental_singleton
def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    #client = create_client(url, key)
    return create_client(url, key)
con = init_connection()
############-------END OF CONNECTION WITH DATABASE/SUPABASE--------#######

##################------------------SIDEBAR-------------------############
with st.sidebar.expander("DELETE USER", expanded=False):
    st.error("Warning this is pernament")
    with st.form("delete user"):
        id_to_delete = st.number_input("Type ID of user to delete", value=0, step=1)
        
        verify_delete_text_input = st.text_input("Type 'Delete' in the field above to proceed")
        id_to_delete_button = st.form_submit_button("Delete User")

    if id_to_delete_button and verify_delete_text_input=="Delete":
        def delete_entry_from_jumps_statistics_table(supabase):
            query=con.table("jumps_statistics_table").delete().eq("id", id_to_delete).execute()
            return query
        query = delete_entry_from_jumps_statistics_table(con)
        # Check if list query.data is empty or not
        if query.data:
            def main():
                delete_entry = delete_entry_from_jumps_statistics_table(con)
            main()
            st.success('Thank you! This entry has been deleted from database!')
        else:
            st.warning("There is no entry with this id to delete!")
############--------END OF SIDEBAR-----------#################

######----START OF THE APP------##########
def select_all_jumps_statistics_table():
    query=con.table("jumps_statistics_table").select("*").execute()
    df = pd.DataFrame(query.data)
    df['date_of_trial'] = pd.to_datetime(df['date_of_trial'])
    df["impulse_bw_duration"] = df["user_time_input_max_jumps_table"] - df["user_time_input_min_jumps_table"]
    df["take_off_to_landing"] = df["landing_time"] - df["user_time_input_max_jumps_table"]
    df['force_div_weight'] = df["force_mean"] / df["weight"]
    # Reranem the columns:

    df.rename(columns = {'id' :'ID',
                        'created_at':'Created at',
                        'fullname':'Fullname', 
                        'age':'Age',
                        'height':'Height',
                        'weight':'Weight',
                        'type_of_trial':'Type of trial',
                        'filename':'Filename',
                        'filepath':'Filepath',
                        'occupy':'Occupy',
                        'jump':'Jump (Impulse)',
                        'rms_1_mean':'RMS 1 mean',
                        'rms_2_mean':'RMS 2 mean',
                        'rms_3_mean':'RMS 3 mean',
                        'force_mean':'Force mean',
                        'force_max':'Force max',
                        'rms_1_norm':'RMS 1 norm',
                        'rms_2_norm':'RMS 2 norm',
                        'rms_3_norm':'RMS 3 norm',
                        'force_min':'Force min',
                        'user_time_input_min_jumps_table':'From time',
                        'user_time_input_max_jumps_table':'Till time',
                        'landing_time':'Landing time',
                        'force_sum':'Force sum',
                        'jump_depending_time_in_air':'Jump (in air)',
                        'user_time_input_start_try_time':'Start trial time',
                        'user_time_input_rfd_from_time':'From rfd time',
                        'user_time_input_rfd_till_time':'Till rfd time',
                        'impulse_bw_duration':'Impulse bw duration',
                        'take_off_to_landing':'Time in air',
                        'force_div_weight':'Force / Weight',
                        'rsi':'rsi',
                        'impulse_bw' : 'Impulse body weight',
                        'impulse_grf' : 'Impulse gravity',
                        'date_of_trial' : 'Date of trial'
                        }, inplace = True)
    df['duration_from_till_time'] = df['Till time'] - df['From time']
    df['Duration'] = df['From time'] - df['Start trial time']

    return df

@st.cache_data
def calculate_kpis(df: pd.DataFrame) -> List[float]:
    total_trials = len(df)
    total_cmj_trials = len(df["Type of trial"][df["Type of trial"] == "CMJ"])
    total_dj_trials = len(df["Type of trial"][df["Type of trial"] == "DJ"])
    total_sj_trials = len(df["Type of trial"][df["Type of trial"] == "SJ"])
    total_iso_trials = len(df["Type of trial"][df["Type of trial"] == "ISO"])
    unique_users = df["Fullname"].nunique()
    return [total_trials,unique_users, total_cmj_trials, total_dj_trials, total_sj_trials,total_iso_trials]

def display_kpi_metrics(kpis: List[float], kpi_names: List[str]):
    st.write("#### Key Performance Indicators (KPIs) Metrics ")
    for i, (col, (kpi_name, kpi_value)) in enumerate(zip(st.columns(6), zip(kpi_names, kpis))):
        col.metric(label=kpi_name, value=kpi_value)

def display_sidebar(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    st.sidebar.header("Filters")
    start_date = pd.Timestamp(st.sidebar.date_input("Start date", df['Date of trial'].min().date()))
    end_date = pd.Timestamp(st.sidebar.date_input("End date", df['Date of trial'].max().date()))
    unique_fullnames = df['Fullname'].unique()
    options_unique_fullnames =[" "] + [unique_fullnames[i] for i in range (0, len(unique_fullnames)) ]
    select_user = st.sidebar.selectbox("Select user:", options = options_unique_fullnames)
    select_type_of_trial = st.sidebar.selectbox("Select type of trial:", options = (' ', 'CMJ', 'SJ', 'DJ', 'ISO'))

    return select_user, select_type_of_trial, start_date, end_date


def display_charts(df, select_user, select_type_of_trial, start_date, end_date):
    st.write("#")
    counts=[]
    col1, col2 = st.columns([1,2], gap="Large")
    with col1:
        st.write("##### Counts for each trial for all date period:")
        fig = px.pie(df, names='Type of trial')
        fig.update_traces(hoverinfo='label+percent', textinfo='label+percent+value')
        fig.update_layout(
            #margin=dict(l=80, r=20, t=20, b=20),
            #paper_bgcolor="LightSteelBlue",
        )
        st.plotly_chart(fig, use_container_width=True)  
    with col2:
        trials_count_per_date = df['Date of trial'].value_counts()[0]
        st.write("##### Trials per Dates for all date period:")
        for idx, name in enumerate(df['Date of trial'].value_counts()): 
            counts.append(df['Date of trial'].value_counts()[idx])
            #st.write(df['Date of trial'].value_counts()[idx]) 
        
        fig = px.bar(df, x=df["Date of trial"].unique() , y=counts, width=900, height=500)
        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        fig.update_xaxes(rangemode='tozero', showgrid=False)
        fig.update_yaxes(rangemode='tozero', showgrid=True)
        st.plotly_chart(fig, use_container_width=True)

    df = df[(df['Date of trial'] > start_date) & (df['Date of trial'] <= end_date)]

    st.write("##### Display data based on time duration between",start_date, " and", end_date )
    st.write("#")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("##### Top 5 Testers by count")
        top_testers = df['Fullname'].value_counts().nlargest(n=5)
        st.write(top_testers)
        #st.metric(label='Top TEsters', value = top_testers, delta = " ")

    with col2:
        st.write("##### Top 5 Jump Heights by testers")
        top_jumps = df[['Fullname', 'Jump (Impulse)']].nlargest(n=5, columns=['Jump (Impulse)'])
        st.write(top_jumps)

    with col4:
        st.write("##### Top 5 rsi by testers:")
        top_rsi = df[['Fullname', 'rsi']].nlargest(n=5, columns=['rsi'])
        st.write(top_rsi)
    with col3:
        st.write("##### Top 5 Force peaks by testers")
        top_force_max = df[['Fullname', 'Force max']].nlargest(n=5, columns=['Force max'])
        st.write(top_force_max)
    

    if select_user != " " and select_type_of_trial != " ":
        st.write("Dataset for", select_user, )
        st.write(df[df['Fullname']==select_user].reset_index())
        st.write("#")
        df_corr = df[df['Fullname'] == select_user].reset_index().copy()
        df_corr = df_corr[['Duration', 'Force max', 'Force mean',  'Force sum', 'Jump (Impulse)' ]]
        df_corr = df_corr[(df['Date of trial'] > start_date) & (df['Date of trial'] <= end_date)]
        #st.write(df_corr.corr()['Jump (Impulse)'])
        col1, col2 = st.columns(2, gap="Large")
        
        with col2:
            st.write("###### Correlation between variables and Jump (Impulse)")
            corr_matrix = df_corr.corr()
            fig = plt.figure(figsize=(15, 10))
            sns.heatmap(corr_matrix, 
                        
                        annot=True, 
                        linewidths=0.5, 
                        fmt= ".2f", 
                        cmap="YlGnBu")
            st.pyplot(fig)
        with col1:
            
            fig = px.bar(df[df['Fullname'] == select_user].reset_index(),  y=df['Jump (Impulse)'][df['Fullname'] == select_user][df['Type of trial']==select_type_of_trial][df["Date of trial"] >= start_date][df["Date of trial"] <= end_date], 
                    title="Jump Heights by type of trial Over Time", width=900, height=500)
            fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            fig.update_xaxes(rangemode='tozero', showgrid=False)
            fig.update_yaxes(rangemode='tozero', showgrid=True)
            st.plotly_chart(fig, use_container_width=True)

def correlation(df):
    
    df_corr = df[['Jump (Impulse)', 'Age', 'Weight', 'Height', 'Duration']].copy() #, 'Impulse body weight','Impulse gravity','Force max', 'Force mean',  'Force sum', 'Jump (Impulse)', 'duration_from_till_time', 'duration_from_start_trial_to_from_time']].copy()
    corr_matrix_jump = df_corr.corr()['Jump (Impulse)']
    corr_matrix_jump
    st.write("The calculation of the sample covariance is as follows:")
    #cov(X, Y) = (sum (x - mean(X)) * (y - mean(Y)) ) * 1/(n-1)

    # Y = Jump (Impulse) , 
    #cov_x_y = 
    dfcov = df[['Duration', 'Jump (Impulse)']][df['Type of trial']=='CMJ']
    st.write("dfcov", dfcov.cov())
    covariance = cov(dfcov['Duration'], dfcov['Jump (Impulse)'])
    st.write("covariance", covariance)

    # seed random number generator
    seed(1)
    # prepare data
    data1 = 20 * randn(1000) + 100
    data2 = data1 + (10 * randn(1000) + 50)
    # calculate Pearson's correlation
    
    corr, _ = pearsonr(dfcov['Duration'], dfcov['Jump (Impulse)'])
    st.write('Pearsons correlation: %.3f' % corr)

    corr, _ = spearmanr(dfcov['Duration'], dfcov['Jump (Impulse)'])
    st.write('Spearmans correlation: %.3f' % corr)

def main():
    # call function to load all the data from database:
    df = select_all_jumps_statistics_table()
    st.title(":bar_chart: Trials Dashboard")
    
    # call function for display the sidebar:
    select_user, select_type_of_trial,start_date, end_date = display_sidebar(df)

    # # call function for displaying the charts:
    # filtered_data = df.copy()
    # filtered_data = filter_data(filtered_data, 'Fullname', unique_fullnames)
    # filtered_data = filter_data(filtered_data, 'Fullname', select_user)
    # filtered_data = filter_data(filtered_data, 'Type of trial', select_type_of_trial)

    kpis = calculate_kpis(df)
    kpi_names = ["Total Trials", "Total Unique Users", "Total CMJ Trials", "Total DJ Trials", "Total SJ Trials", "Total ISO Trials"]
    
    display_kpi_metrics(kpis, kpi_names)
    display_charts(df, select_user, select_type_of_trial, start_date, end_date)
    # st.write("#")
    # st.write("##### Display corellation depending on Jump:")
    # correlation(df)
    

if __name__ == '__main__':
    main()
