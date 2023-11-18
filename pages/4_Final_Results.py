import os
import streamlit as st
import pandas as pd
from supabase import create_client, Client
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
       
###############-------SET PAGE CONFIGURATION--------##########
st.set_page_config(
    page_title="Tefaa Metrics",
    page_icon="🧊",
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
    return query
query = select_all_jumps_statistics_table()
df = pd.DataFrame(query.data)
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
                     'jump':'Jump (Impluse)',
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
                     'impulse_bw_duration':'Impluse bw duration',
                     'take_off_to_landing':'Time in air',
                     'force_div_weight':'Force / Weight'
                     }, inplace = True)

st.write("## Final Results of trials:")
# st.write("""0) df_brushed['Impulse_grf'] = df_brushed['Force'] * (1/1000)
# 1) impulse_grf = df_brushed['Impulse_grf'].sum()
# 2) impulse_bw_duration = (user_time_input_max_jumps_table - user_time_input_min_jumps_table) / 1000
# 3) impulse_bw = url_list[0]['weight'] * 9.81 * impulse_bw_duration
# 4) velocity_momentum1 = (impulse_grf - impulse_bw) / url_list[0]['weight']
# 5) jump_depending_impluse = (velocity_momentum1 ** 2) / (9.81 * 2) """)
fullname_search = " "
if not df.empty:
    #st.dataframe(df.loc[:, ~df.columns.isin(['Filename', 'Filepath'])].sort_values('Created at', ascending=False),use_container_width=True)
    col1, col2, col3 = st.columns([3,2,2])
    with col2:
        type_of_trial_search = st.selectbox("Επέλεξε ίδος προσπάθειας  " , options = (" ", "CMJ", "DJ", "ISO", "SJ"))
    with col3:
        occupy_search = st.text_input("Occupy:")
    with col1:
        unique_fullnames = df['Fullname'].unique()
        options =[" "] + [unique_fullnames[i] for i in range (0, len(unique_fullnames)) ]
        fullname_search = st.selectbox("Αναφορά σε Χρήστη  " , options = options)
    # conditions depending on searches input fields:
    if not occupy_search and fullname_search == " " and type_of_trial_search == " ":
        st.dataframe(df.loc[:, ~df.columns.isin(['Filename', 'Filepath'])].sort_values('Created at', ascending=False), use_container_width=True)
    elif fullname_search and not occupy_search and type_of_trial_search == " ":
        st.dataframe(df.loc[:, ~df.columns.isin(['Filename', 'Filepath'])].sort_values('Created at', ascending=False)[df['Fullname']== fullname_search], use_container_width=True)
    elif occupy_search and fullname_search == " " and type_of_trial_search == " ":
        st.dataframe(df.loc[:, ~df.columns.isin(['Filename', 'Filepath'])].sort_values('Created at', ascending=False)[df['Occupy']== occupy_search], use_container_width=True)
    elif type_of_trial_search and fullname_search == " " and not occupy_search:
        st.dataframe(df.loc[:, ~df.columns.isin(['Filename', 'Filepath'])].sort_values('Created at', ascending=False)[df['Type of trial']== type_of_trial_search], use_container_width=True)
    elif fullname_search and occupy_search and type_of_trial_search == " ":
        st.dataframe(df.loc[:, ~df.columns.isin(['Filename', 'Filepath'])].sort_values('Created at', ascending=False)[(df['Fullname'] == fullname_search) & (df['Occupy'] == occupy_search)], use_container_width=True)
    elif fullname_search and type_of_trial_search and not occupy_search:
        st.dataframe(df.loc[:, ~df.columns.isin(['Filename', 'Filepath'])].sort_values('Created at', ascending=False)[(df['Fullname'] == fullname_search) & (df['Type of trial'] == type_of_trial_search)], use_container_width=True)
    elif occupy_search and type_of_trial_search and fullname_search == " ":
        st.dataframe(df.loc[:, ~df.columns.isin(['Filename', 'Filepath'])].sort_values('Created at', ascending=False)[(df['Occupy'] == occupy_search) & (df['Type of trial'] == type_of_trial_search)], use_container_width=True)
    elif fullname_search and occupy_search and type_of_trial_search:
        st.dataframe(df.loc[:, ~df.columns.isin(['Filename', 'Filepath'])].sort_values('Created at', ascending=False)[(df['Occupy'] == occupy_search) & (df['Fullname'] == fullname_search) & (df['Type of trial'] == type_of_trial_search)], use_container_width=True)        
else:
    st.write("There are no entries in the database! Please insert first!")

st.write("##### List of unique persons who made trial: ")
col1,col2,col3 = st.columns(3)
with col1:
    st.write(df.Fullname.unique())
with col2:
    st.write("Highest jump.", df["Jump (Impluse)"].max())
with col3:
    st.write("Max force", df["Force max"].max())


col1,col2,col3 = st.columns(3, gap="large")
columns = [" "] + [df.columns[i] for i in range (0, len(df.columns))]
with col1:
    fig1 = plt.figure(figsize=(8,7))
    select_person1 = st.selectbox('Select the person 1',
    options = options)
    select_column1 = st.selectbox('Select Column',
    options = columns)
    if select_column1 != " ":
        df[df["Fullname"] == select_person1][select_column1].plot.line()
        plt.xlabel("Times of trial")
        plt.ylabel(select_column1)
        st.pyplot(fig1)

with col2:
    fig2 = plt.figure(figsize=(8,7))
    select_person2 = st.selectbox('Select the person 2',
    options = options)
    df[df["Fullname"] == select_person2]["Force / Weight"].plot.line()
    plt.xlabel("Times of trial")
    plt.ylabel("Jump (Impluse) in meter")
    st.pyplot(fig2)
    st.write("#")
with col3:
    st.write("")
    #co1 = df[df["Fullname"] == "Tziavras Dimitris"]["Force mean"] + df[df["Fullname"] == "Tziavras Dimitris"]["Weight"] * 9.82
    chart_data = pd.DataFrame(
        {
            "co1": df["Time in air"],
            "col2": df["Jump (in air)"],
            #"col3": np.random.choice(["A", "B", "C"], 45),
        }
    )
    st.area_chart(chart_data, x="co1", y="col2")


