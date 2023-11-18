####----IMPORT PACKAGES----#####
import os
import streamlit as st
import pandas as pd
import numpy as np
from supabase import create_client, Client
import altair as alt
import biosignalsnotebooks as bsnb
import plotly.graph_objects as go
import sympy as sy
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
# biosignalsnotebooks own package for loading and plotting the acquired data
import biosignalsnotebooks as bsnb
# Scientific packages
from numpy import arange, sin, pi
from numpy.random import randn

############## ############## PAGE 3 CALCULATE RESULTS ############# ############# ############## ########################
st.set_page_config(
    page_title="Tefaa Metrics",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Make the connection with Supabase - Database:
#@st.experimental_singleton
# set the variable g:
g = 9.81
# create the connection with supabase database:
def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    #client = create_client(url, key)
    return create_client(url, key)
con = init_connection()

#########################----SIDEBAR------##########################
st.title("Calculate Results")
with st.sidebar.expander("DELETE USER", expanded=False):
    st.error("Warning this is pernament")
    with st.form("delete user"):
        id_to_delete = st.number_input("Type ID of user to delete", value=0, step=1)
        
        verify_delete_text_input = st.text_input("Type 'Delete' in the field above to proceed")
        id_to_delete_button = st.form_submit_button("Delete User")

    if id_to_delete_button and verify_delete_text_input=="Delete":
        def delete_entry_from_jumps_table(supabase):
            query=con.table("jumps_table").delete().eq("id", id_to_delete).execute()
            return query
        query = delete_entry_from_jumps_table(con)
        # Check if list query.data is empty or not
        if query.data:
            def main():
                delete_entry = delete_entry_from_jumps_table(con)
            main()
            st.success('Thank you! This entry has been deleted from database!')
        else:
            st.warning("There is no entry with this id to delete!")
#########################----END OF SIDEBAR------##########################


#########################----MAIN AREA------##########################
# Fetch and display the whole table with entries:
url_list=[]
# Get all the trials from table, create a dataframe and display it:
with st.expander("List of all entries from the database.", expanded=True):
    st.caption("Use the below search fields to filter the datatable!")
    #@st.experimental_memo(ttl=300)
    # function: query to get the data of table:
    def select_all_from_jumps_table():
        query=con.table("jumps_table").select("*").execute()
        return query
    query = select_all_from_jumps_table()
    # dataframe with the results of table (query):
    df_jumps_table = pd.DataFrame(query.data)
    fullname_search = " "
    # create search fields to search on the dataframe:
    if not df_jumps_table.empty:
        df_jumps_table.columns = ['ID', 'Created At', 'Fullname', 'Email', 'Occupy', 'Type of Trial', 'Filename', 'Filepath', 'Height', 'Weight', 'Age', 'Instructor', 'Drop Height']
        col1, col2, col3 = st.columns([3,2,2])
        with col2:
            type_of_trial_search = st.selectbox("Επέλεξε ίδος προσπάθειας  " , options = (" ", "CMJ", "DJ", "ISO", "SJ"))
        with col3:
            occupy_search = st.text_input("Occupy:")
        with col1:
            unique_fullnames = df_jumps_table['Fullname'].unique()
            options =[" "] + [unique_fullnames[i] for i in range (0, len(unique_fullnames)) ]
            fullname_search = st.selectbox("Αναφορά σε Χρήστη  " , options = options)
        # conditions depending on searches input fields:
        if not occupy_search and fullname_search == " " and type_of_trial_search == " ":
            st.dataframe(df_jumps_table[['ID', 'Created At', 'Fullname', 'Occupy', 'Type of Trial', 'Filename', 'Height', 'Weight', 'Age', 'Instructor']].sort_values('Created At', ascending=False))
        
        elif fullname_search and not occupy_search and type_of_trial_search == " ":
            st.dataframe(df_jumps_table[df_jumps_table['Fullname']== fullname_search], use_container_width=True)

        elif occupy_search and not fullname_search and not type_of_trial_search:
            st.dataframe(df_jumps_table[df_jumps_table['Occupy']== occupy_search], use_container_width=True)

        elif type_of_trial_search and fullname_search == " " and not occupy_search:
            st.dataframe(df_jumps_table[df_jumps_table['Type of Trial']== type_of_trial_search])

        elif fullname_search and occupy_search and not type_of_trial_search:
            df_jumps_table[(df_jumps_table['Fullname'] == fullname_search) & (df_jumps_table['Occupy'] == occupy_search)]

        elif fullname_search and type_of_trial_search and not occupy_search:
            df_jumps_table[(df_jumps_table['Fullname'] == fullname_search) & (df_jumps_table['Type of Trial'] == type_of_trial_search)]
        
        elif occupy_search and type_of_trial_search:
            df_jumps_table[(df_jumps_table['Occupy'] == occupy_search) & (df_jumps_table['Type of Trial'] == type_of_trial_search)]
        
        elif fullname_search and occupy_search and type_of_trial_search:
            df_jumps_table[(df_jumps_table['Occupy'] == occupy_search) & (df_jumps_table['Fullname'] == fullname_search) & (df_jumps_table['Type of Trial'] == type_of_trial_search)]
    else:
        st.write("There are no entries in the database! Please insert first!")

with st.sidebar.form("Type the ID of your link:", clear_on_submit=False):   
    url_id_number_input = st.number_input("Type the ID of your prerferred trial and Press Calculate Results:",value = 0,step= 1)
    id_submitted = st.form_submit_button("Calculate Results")
    # Querry to find the data row of specific ID
    if url_id_number_input:
        def select_filepath_from_specific_id():
            query=con.table("jumps_table").select("*").eq("id", url_id_number_input).execute()
            return query
        query = select_filepath_from_specific_id()  
        # Make a list with all values from database depending on the condition. 
        url_list =  query.data
        # List with values depending on the querry
        if url_list:
            url = url_list[0]['filepath'].replace(" ", "%20")
            st.write("Person ID:", url_list[0]['id'])
        else:
            st.write("There is no entry with this ID")


# main function to get all the data from trial-file, depending on the id of the user:
def get_data():
    if url_list:
        storage_options = {'User-Agent': 'Mozilla/5.0'}
        df = pd.read_csv(url_list[0]['filepath'].replace(" ", "%20"), storage_options=storage_options)
        # Define Header columns:
        columns_count = len(df.axes[1])
        # Define next columns: 
        df['pre_pro_signal_EMG_1'] = 0
        df['pre_pro_signal_EMG_2'] = 0
        df['pre_pro_signal_EMG_3'] = 0
        df['RMS_1'] = float("nan")
        df['RMS_2'] = float("nan")
        df['RMS_3'] = float("nan")
        df['Acceleration'] = float("nan")
        df['Start_Velocity'] = float("nan")
        df['Velocity'] = float("nan")
        df['Rows_Count'] = df.index
        low_cutoff = 10 # Hz
        high_cutoff = 450 # Hz
        frequency = 1000
        # Calculate The Column Force
        df['Force'] = df['Mass_Sum'] * 9.81
        #IF type_of_trial = ISO , αφαιρεσε το σωματικο βαρος.
        if url_list[0]['type_of_trial'] == "ISO":
            df['Force'] = df['Force'] - url_list[0]['weight'] * 9.81
        # Calculate Acceleration, Velocity for CMJ and SJ Trials:
        if url_list[0]['type_of_trial'] == "CMJ" or url_list[0]['type_of_trial'] == "SJ":
            df['Acceleration'] = (df['Force'] / url_list[0]['weight']) - 9.81
            df['Start_Velocity'] = df.Acceleration.rolling(window=2,min_periods=1).mean()*0.001
            df['Velocity'] = df.Start_Velocity.rolling(window=999999,min_periods=1).sum()

        #####------THIS IS ALL FOR EMG TO RMS 1------########
        if 'Col_9' in df.columns:
            # [Baseline Removal] Convert Raw Data EMG to EMG
            df['Col_9_to_converted'] = (((df['Col_9']/ 2 ** 16) - 1/2 ) * 3 ) / 1000
            df['Col_9_to_converted'] = df['Col_9_to_converted'] *1000
            pre_pro_signal_1 = df['Col_9_to_converted'] - df["Col_9_to_converted"].mean()
            # Application of the signal to the filter. This is EMG1 after filtering
            pre_pro_signal_1= bsnb.aux_functions._butter_bandpass_filter(pre_pro_signal_1, low_cutoff, high_cutoff, frequency)
            df['pre_pro_signal_EMG_1'] = pre_pro_signal_1**2
            #This is RMS per 100
            df['RMS_1'] = df.pre_pro_signal_EMG_1.rolling(window=100,min_periods=100).mean()**(1/2)
        
        ######------THIS IS ALL FOR EMG TO RMS 2------##########
        if 'Col_10' in df.columns: 
            df['Col_10_to_converted'] = (((df['Col_10']/ 2 ** 16) - 1/2 ) * 3 ) / 1000
            df['Col_10_to_converted'] = df['Col_10_to_converted'] *1000
            pre_pro_signal_2 = df['Col_10_to_converted'] - df["Col_10_to_converted"].mean()
            # Application of the signal to the filter. This is EMG1 after filtering
            pre_pro_signal_2= bsnb.aux_functions._butter_bandpass_filter(pre_pro_signal_2, low_cutoff, high_cutoff, frequency)
            df['pre_pro_signal_EMG_2'] = pre_pro_signal_2**2
            #This is RMS per 100
            df['RMS_2'] = df.pre_pro_signal_EMG_2.rolling(window=100,min_periods=100).mean()**(1/2)

        #######-----THIS IS ALL FOR EMG TO RMS 3-----#########
        if 'Col_11' in df.columns:
            df['Col_11_to_converted'] = (((df['Col_11']/ 2 ** 16) - 1/2 ) * 3 ) / 1000
            df['Col_11_to_converted'] = df['Col_11_to_converted'] *1000
            pre_pro_signal_3 = df['Col_11_to_converted'] - df["Col_11_to_converted"].mean()
            # Application of the signal to the filter. This is EMG1 after filtering
            pre_pro_signal_3= bsnb.aux_functions._butter_bandpass_filter(pre_pro_signal_3, low_cutoff, high_cutoff, frequency)
            df['pre_pro_signal_EMG_3'] = pre_pro_signal_3**2
            #This is RMS per 100
            df['RMS_3'] = df.pre_pro_signal_EMG_3.rolling(window=100,min_periods=100).mean()**(1/2)
        
        df['Force'] = bsnb.lowpass(df['Force'], 20, order=2, use_filtfilt=True)

        return df

if url_list:
    df = get_data()
    #Find standard deviation
    for i in range(0,450):
         xi_xmean = (((df.loc[i, 'Force'] - df.loc[1:450,'Force'].mean()))*2)
    xi_xmean_sum = xi_xmean.sum()
    std = ( ( xi_xmean_sum ** 2 ) / ( 2000 -1 ) ) * ( 1 / 2)
    
    ######------FIND TIMES FOR CMJ TRIAL--------#######
    if url_list[0]['type_of_trial'] == "CMJ":
        # Find Take Off Time: 
        for i in range (0, len(df.index)):
            if df.loc[i,'Force'] < 2 :
                take_off_time = i
                break

        # Find Landing Time:
        for i in range (take_off_time, len(df.index)):
            if df.loc[i,'Force'] > 80:
                landing_time = i - 1
                break
        # Find Start Try Time
        for i in range(0,take_off_time):
            if df.loc[i,'Force'] < (df['Force'].mean() - 80):
                start_try_time = i
                break
        closest_to_zero_velocity = df.loc[start_try_time:take_off_time,'Velocity'].sub(0).abs().idxmin()
        closest_to_average_force_1st = (df.loc[start_try_time:closest_to_zero_velocity,'Force']-df['Force'].mean()).sub(0).abs().idxmin()
    ######------END OF FIND TIMES FOR CMJ TRIAL--------#######

        # calculate velocity landing:
        velocity_landing =  - ( 2 * g * url_list[0]['drop_height'] ) ** (1/2)
        # calculate force_empty which is the mean of some initial force values:
        force_empty = (df.loc[0:300, 'Force']).mean()     
        # algorithm to find the velocity: Αλγοριθμος για να βρω την ταχυτητα
        for i in range (0,len(df)):
            if df.loc[i, 'Force'] < force_empty * g * 1.05 :
                df.loc[i,'Velocity'] = velocity_landing
            
            # if df.loc[i, 'Force'] >= force_empty * g * 1.05 and df.loc[i, 'Force'] <= url_list[0]['weight'] * g :
            #     df.loc[i, 'Velocity'] = velocity_landing + df.loc[i-1:i, 'Net_Force'].mean() * 0.01 /  ( url_list[0]['weight'] * g )
            else :
                df.loc[i, 'Velocity'] = df.loc[i-1, 'Velocity'] + df.loc[i-1:i, 'Net_Force'].mean() * 0.01 / ( url_list[0]['weight'] * g )

        # st.write("force_empty * 9.81 * 1.05",force_empty * 9.81 * 1.05)
        # st.write("df.loc[3000, 'Force']", df.loc[3000, 'Force'])
        # st.write("velocity landing",velocity_landing)
        # st.write("url_list[0]['weight'] * g ",url_list[0]['weight'] * g )
        
        # Βρισκω την πρωτη φορα κοντα στο μηδεν ταχυτητα
        closest_to_zero_velocity = df.loc[start_try_time:take_off_time,'Velocity'].sub(0).abs().idxmin()
        #closest_to_zero_force = df.loc[start_try_time:len(df),'Force'].sub(0).abs().idxmin()

        # Βρισκω το Net Impluse, Απο το διαστημα 1ης φορας ταχυτητας μηδεν μεχρι 1ης φορας δυναμης μηδεν, βρισκω μεσο ορο των τιμων της στηλης Net Force
        net_impluse = df.loc[closest_to_zero_velocity:take_off_time, 'Net_Force'].mean()
        st.write('net_impluse',net_impluse)
        # Βρισκω την διαφορα χρονου 1η φορα δυναμη μηδεν πλην 1η φορα ταχυτητα μηδεν
        concentric_time = take_off_time - closest_to_zero_velocity 
        closest_to_zero_velocity
        # Βρισκω Velocity Take off
        velocity_take_off = (concentric_time / 1000 * net_impluse) / url_list[0]['weight']
        # Βρισκω το αλμα του DJ βασισμενο σε Velocity Take off
        jump_depending_take_off_velocity = ( velocity_take_off ** 2 ) / ( 2 * g ) 
        jump_depending_take_off_velocity
        net_impluse
        # calculate one new variable (contact_time), which is from the time the user steps on the platform until the time he does not:
        contact_time = take_off_time - start_try_time

        st.write("Velocity Take Off from time take of ",df.loc[take_off_time, 'Velocity'])
        st.write("Velocity Take Off from equation ", velocity_take_off)
