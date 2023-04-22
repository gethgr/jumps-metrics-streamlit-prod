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

############## ############## PAGE 3 CALCULATE RESULTS ############# ############# ############## ########################
st.set_page_config(
    page_title="Tefaa Metrics",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

#Make the connection with Supabase - Database:
#@st.experimental_singleton
g = 9.81
def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    #client = create_client(url, key)
    return create_client(url, key)
con = init_connection()

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


# Fetch and display the whole table with entries:
url_list=[]
with st.expander("List of all entries from the database.", expanded=True):
    st.caption("Use the below search fields to filter the datatable!")
    #uploaded_file = st.file_uploader("Choose a file1")
    #@st.experimental_memo(ttl=300)
    def select_all_from_jumps_table():
        query=con.table("jumps_table").select("*").execute()
        return query
    query = select_all_from_jumps_table()


    df_jumps_table = pd.DataFrame(query.data)
    fullname_search = " "
    if not df_jumps_table.empty:
        df_jumps_table.columns = ['ID', 'Created At', 'Fullname', 'Email', 'Occupy', 'Type of Trial', 'Filename', 'Filepath', 'Height', 'Weight', 'Age', 'Instructor', 'Drop Height']
        col1, col2, col3 = st.columns([3,2,2])
        with col2:
            type_of_trial_search = st.selectbox("Επέλεξε ίδος προσπάθειας  " , options = (" ", "CMJ", "DJ", "ISO"))
        with col3:
            occupy_search = st.text_input("Occupy:")
        with col1:
            unique_fullnames = df_jumps_table['Fullname'].unique()
            #st.write(df_jumps_table['Fullname'].unique())
            options =[" "] + [unique_fullnames[i] for i in range (0, len(unique_fullnames)) ]
            #i = df_jumps_table.loc[i,['Fullname']]]

            fullname_search = st.selectbox("Αναφορά σε Χρήστη  " , options = options)
        
        

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

        #url_id_number_input = st.number_input("Type the ID of the person you want to calculate results of the current trial.",value=0,step=1)


        # In this form, you type the id of the person to calculate speicific trial.
        
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


#@st.cache(allow_output_mutation=True)
def get_data():
    if url_list:
        storage_options = {'User-Agent': 'Mozilla/5.0'}
        df = pd.read_csv(url_list[0]['filepath'].replace(" ", "%20"), storage_options=storage_options)
        # #Define Header columns
        columns_count = len(df.axes[1])
        
        #Define next columns 
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

        #pm = df['Mass_Sum'].mean()

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

           # gia to 6o deuterolepto , vriskw thn mesh timh tou 5o k 6o sthn epitaxinsi kai to prosthetw sthn taxitita tou 5ou 

        # THIS IS ALL FOR EMG TO RMS 1
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
        
        # THIS IS ALL FOR EMG TO RMS 2
        if 'Col_10' in df.columns: 
            df['Col_10_to_converted'] = (((df['Col_10']/ 2 ** 16) - 1/2 ) * 3 ) / 1000
            df['Col_10_to_converted'] = df['Col_10_to_converted'] *1000
            pre_pro_signal_2 = df['Col_10_to_converted'] - df["Col_10_to_converted"].mean()
            # Application of the signal to the filter. This is EMG1 after filtering
            pre_pro_signal_2= bsnb.aux_functions._butter_bandpass_filter(pre_pro_signal_2, low_cutoff, high_cutoff, frequency)
            df['pre_pro_signal_EMG_2'] = pre_pro_signal_2**2
            #This is RMS per 100
            df['RMS_2'] = df.pre_pro_signal_EMG_2.rolling(window=100,min_periods=100).mean()**(1/2)

        # THIS IS ALL FOR EMG TO RMS 3
        if 'Col_11' in df.columns:
            df['Col_11_to_converted'] = (((df['Col_11']/ 2 ** 16) - 1/2 ) * 3 ) / 1000
            df['Col_11_to_converted'] = df['Col_11_to_converted'] *1000
            pre_pro_signal_3 = df['Col_11_to_converted'] - df["Col_11_to_converted"].mean()
            # Application of the signal to the filter. This is EMG1 after filtering
            pre_pro_signal_3= bsnb.aux_functions._butter_bandpass_filter(pre_pro_signal_3, low_cutoff, high_cutoff, frequency)
            df['pre_pro_signal_EMG_3'] = pre_pro_signal_3**2
            #This is RMS per 100
            df['RMS_3'] = df.pre_pro_signal_EMG_3.rolling(window=100,min_periods=100).mean()**(1/2)
        
        return df


############################################################################################################                

if url_list:
    df = get_data()
    rsi = float("nan")
    ####### ###### ##### FIND TIMES FOR CMJ TRIAL ####### ######### #######
    if url_list[0]['type_of_trial'] == "CMJ":
        # Find Take Off Time: 
        for i in range (0, len(df.index)):
            if df.loc[i,'Force'] < 2:
                take_off_time = i
                break
        # Find Landing Time:
        for i in range (take_off_time, len(df.index)):
            if df.loc[i,'Force'] > 55:
                landing_time = i - 1
                break
        # Find Start Try Time
        for i in range(0,take_off_time):
            if df.loc[i,'Force'] < (df['Force'].mean() - 80):
                start_try_time = i
                break
        closest_to_zero_velocity = df.loc[start_try_time:take_off_time,'Velocity'].sub(0).abs().idxmin()
        closest_to_average_force_1st = (df.loc[start_try_time:closest_to_zero_velocity,'Force']-df['Force'].mean()).sub(0).abs().idxmin()
   
    ####### ###### ##### FIND TIMES FOR SJ TRIAL ####### ######### #######
    if url_list[0]['type_of_trial'] == "SJ":
        for i in range (0, len(df.index)):
            if df.loc[i,'Force'] < 2:
                take_off_time = i
                break
        # Find Landing Time:
        for i in range (take_off_time, len(df.index)):
            if df.loc[i,'Force'] > 55:
                landing_time = i - 1
                break
        for i in range(0,take_off_time):
            if df.loc[i,'Force'] > (df.loc[10,'Force'] + 30):
                start_try_time = i
                break

    ####### ###### ##### FIND TIMES FOR DJ TRIAL ####### ######### #######
    if url_list[0]['type_of_trial'] == "DJ":
        for i in range(len(df.index)):
            if df.loc[i,'Force'] > 27:
                start_try_time = i
                break
        for i in range(start_try_time,len(df.index)):
            if df.loc[i,'Force'] < 5:
                take_off_time = i
                break
        for i in range(take_off_time,len(df.index)):
            if df.loc[i,'Force'] > 35:
                landing_time = i
                break
        
        drop_height = 0.25
        velocity_landing =  - ( 2 * g * drop_height ) * (1/2)
        
        force_empty = (df.loc[0:1000, 'Force'] / 9.81).mean()
        
        st.write(force_empty * g * 1.05)
        
        
        for i in range (0,len(df)):
            if df.loc[i, 'Force'] < force_empty * g * 1.05 :
                df.loc[i,'Velocity'] = velocity_landing
            
            if df.loc[i, 'Force'] > force_empty * g * 1.05 and df.loc[i, 'Force'] < url_list[0]['weight'] * g :
                df.loc[i, 'Velocity'] = velocity_landing + df.loc[i-1:i, 'Force'].mean() * 0.01 /  ( url_list[0]['weight'] * g )

            if df.loc[i, 'Force'] > url_list[0]['weight'] * g :
                df.loc[i, 'Velocity'] = df.loc[i-1, 'Velocity'] + df.loc[i-1:i, 'Force'].mean() * 0.01 / ( url_list[0]['weight'] * g )
        
        
        closest_to_zero_velocity = df.loc[start_try_time:take_off_time,'Velocity'].sub(0).abs().idxmin()
        #closest_to_zero_force = df.loc[start_try_time:len(df),'Force'].sub(0).abs().idxmin()

        concentric_time = take_off_time - closest_to_zero_velocity 


        vel_takeoff = (concentric_time / 1000 * 795) / url_list[0]['weight']


        jump = vel_takeoff ** 2 / 2 * g
        st.write('closest_to_zero_velocity',closest_to_zero_velocity)
        st.write('take_off_time',take_off_time)
        st.write('time', concentric_time)
        st.write('velocit take off', vel_takeoff)
        st.write('alma',jump)
        st.write('landing velocity', velocity_landing)
        



        # df.loc[start_try_time:len(df), 'Acceleration'] = (df.loc[start_try_time:len(df), 'Force'] / url_list[0]['weight']) - 9.81
        # df['Start_Velocity'] = df.Acceleration.rolling(window=2,min_periods=1).mean()*0.001
        # df['Velocity'] = df.Start_Velocity.rolling(window=99999,min_periods=1).sum()

        df.loc[take_off_time, 'Velocity']
    
    with st.expander(("Graph"), expanded=True):
        #### CREATE THE MAIN CHART #####
        fig = go.Figure()
        lines_to_hide = ["RMS_1","RMS_2","RMS_3"]
        # add x and y values for the 1st scatter
        # plot and name the yaxis as yaxis1 values
        fig.add_trace(go.Scatter(
            x=df['Rows_Count'],
            y=df['Force'],
            name="Force",
            line=dict(color="#290baf")
            
        ))
        # add x and y values for the 2nd scatter
        # plot and name the yaxis as yaxis2 values
        fig.add_trace(go.Scatter(
            x=df['Rows_Count'],
            y=df['Velocity'],
            name="Velocity",
            yaxis="y2",
            line=dict(color="#aa0022")
        ))
        
        # add x and y values for the 3rd scatter
        # plot and name the yaxis as yaxis3 values
        fig.add_trace(go.Scatter(
            x=df['Rows_Count'],
            y=df['RMS_1'],
            name="RMS_1",
            yaxis="y3"
        ))
        # add x and y values for the 4th scatter plot
        # and name the yaxis as yaxis4 values
        fig.add_trace(go.Scatter(
            x=df['Rows_Count'],
            y=df['RMS_2'],
            name="RMS_2",
            yaxis="y4",
            line=dict(color="#7b2b2a")
        ))
        fig.add_trace(go.Scatter(
            x=df['Rows_Count'],
            y=df['RMS_3'],
            name="RMS_3",
            yaxis="y5",
            
        ))
        # Create axis objects
        fig.update_layout(
            # split the x-axis to fraction of plots in
            # proportions
            autosize=False,
            title_text="5 y-axes scatter plot",
            #width=1420,
            height=550,
            title_x=0.3,
            margin=dict(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=4
            ),
            hovermode='x',
            plot_bgcolor="#f9f9f9",
            paper_bgcolor='#f9f9f9',
            xaxis=dict(
                domain=[0.125, 0.92],
                linecolor="#BCCCDC",
                showspikes=True, # Show spike line for X-axis
                #Format spike
                spikethickness=2,
                spikedash="dot",
                spikecolor="#999999",
                spikemode="toaxis",
                
                #spikemode= 'toaxis' #// or 'across' or 'marker'      
            ),
            # pass the y-axis title, titlefont, color
            # and tickfont as a dictionary and store
            # it an variable yaxis
            yaxis=dict(
                title="Force",
                titlefont=dict(
                    color="#0000ff"
                ),
                tickfont=dict(
                    color="#0000ff"
                ),
                linecolor="#BCCCDC",
                showspikes=True,
                spikethickness=2,
                spikedash="dot",
                spikecolor="#999999",
                spikemode="toaxis",
                
            ),
            # pass the y-axis 2 title, titlefont, color and
            # tickfont as a dictionary and store it an
            # variable yaxis 2
            yaxis2=dict(
                title="Velocity",
                titlefont=dict(
                    color="#FF0000"
                ),
                tickfont=dict(
                    color="#FF0000"
                ),
                anchor="free",  # specifying x - axis has to be the fixed
                overlaying="y",  # specifyinfg y - axis has to be separated
                side="left",  # specifying the side the axis should be present
                position=0.06,  # specifying the position of the axis

                linecolor="#BCCCDC",
                showspikes=True,
                # spikethickness=2,
                # spikedash="dot",
                # spikecolor="#999999",
                # spikemode="toaxis",

                
                
            ),
            # pass the y-axis 3 title, titlefont, color and
            # tickfont as a dictionary and store it an
            # variable yaxis 3
            yaxis3=dict(
                title="RMS_1",
                titlefont=dict(
                    color="#006400"
                ),
                tickfont=dict(
                    color="#006400"
                ),
                anchor="x",     # specifying x - axis has to be the fixed
                overlaying="y",  # specifyinfg y - axis has to be separated
                side="right" # specifying the side the axis should be present
                #position=0.85
            ),
            
            # pass the y-axis 4 title, titlefont, color and
            # tickfont as a dictionary and store it an
            # variable yaxis 4
            yaxis4=dict(
                title="RMS_2",
                titlefont=dict(
                    color="#7b2b2a"
                ),
                tickfont=dict(
                    color="#7b2b2a"
                ),
                anchor="free",  # specifying x - axis has to be the fixed
                overlaying="y",  # specifyinfg y - axis has to be separated
                side="right",  # specifying the side the axis should be present
                position=0.98  # specifying the position of the axis
            ),
            yaxis5=dict(
                title="RMS_3",
                titlefont=dict(
                    color="#ffbb00"
                ),
                tickfont=dict(
                    color="#ffbb00"
                ),
                anchor="free",  # specifying x - axis has to be the fixed
                overlaying="y",  # specifyinfg y - axis has to be separated
                side="left",  # specifying the side the axis should be present
                position=0.00  # specifying the position of the axis
            )
        )
        # Update layout of the plot namely title_text, width
        # and place it in the center using title_x parameter
        # as shown
        large_rockwell_template = dict(
            layout=go.Layout(title_font=dict(family="Rockwell", size=24))
        )
        
        #     #template=large_rockwell_template
        #     # barmode='group',
        #     #hovermode='x',#paper_bgcolor="LightSteelBlue"   
        # )
        
        fig.update_xaxes(
            
            rangeslider_visible=True,
            
        )
        # This is to hide by default some line
        fig.for_each_trace(lambda trace: trace.update(visible="legendonly") 
                        if trace.name in lines_to_hide else ())
        st.plotly_chart(fig,use_container_width=True)

    ###### ##### ##### DISPLAY Important Times of the graph: ##### ###### ######
    if url_list[0]['type_of_trial'] != 'ISO':
        st.write("#")
        st.write("**Helpfull information about the times of the graph after the start!:**")
    else:
        st.write("#")
        st.write("**Input below fields to calculate results in specific time period:**")

    c1, c2, c3 = st.columns(3)
    with c1:
        if url_list[0]['type_of_trial'] == 'CMJ':
            st.write(" Velocity closest to zero is at:", closest_to_zero_velocity)
        if url_list[0]['type_of_trial'] == "SJ" or url_list[0]['type_of_trial'] == "DJ":
            st.write("The trial starts at:", start_try_time)
    with c2:
        if url_list[0]['type_of_trial'] == 'CMJ' or url_list[0]['type_of_trial'] == 'SJ' or url_list[0]['type_of_trial'] == 'DJ':
            st.write(" Take Off Time is at:", take_off_time)
    with c3:
        if url_list[0]['type_of_trial'] == 'CMJ' or url_list[0]['type_of_trial'] == 'SJ' or url_list[0]['type_of_trial'] == 'DJ':
            st.write(" Landing Time is at:", landing_time)
    
    ###### ###### ###### SELECT TIME PERIOD OF DATASET #### ###### ###### ######## #####
    col1, col2 = st.columns(2)
    r=0  
    with st.form("Form for times",clear_on_submit = False):
        c1, c2, c3, c4, c5= st.columns(5)
        with c1:        
            user_time_input_min_jumps_table = st.number_input("From Time", value=0, step=1, help="Τhe beginning of the desired time interval.")

        with c2:
            user_time_input_max_jumps_table = st.number_input("Till Time", value=0, step=1, help="The end of the desired time interval." )#int(df.index.max()))
 
        with c3:
            if url_list[0]['type_of_trial'] != "ISO":
                rms_1_iso = st.number_input("ISO RMS 1", help="The first ISO RMS Value." )
            else:
                from_time_rfd_iso = st.number_input("From Time for RFD", value=0, step=1, help="Τhe beginning of the desired time interval for RFD." )
            
        with c4:
            if url_list[0]['type_of_trial'] != "ISO":
                rms_2_iso = st.number_input("ISO RMS 2", help="The second ISO RMS Value.")
            else:
                till_time_rfd_iso = st.number_input("Till Time for RFD", value=0, step=1, help="Τhe end of the desired time interval for RFD." )
            
        with c5:
            if url_list[0]['type_of_trial'] != "ISO":
                rms_3_iso = st.number_input("ISO RMS 3", help="The third ISO RMS Value.")
            
        brushed_submitted = st.form_submit_button("Calculate results")

    df_brushed = df[(df.index >= user_time_input_min_jumps_table) & (df.index < user_time_input_max_jumps_table)]
    jump_depending_impluse = float("nan")

    # Find the Jump depending on time in Air and on Take Off Velocity for CMJ & SJ Trial:
    if url_list[0]['type_of_trial'] == "CMJ" or url_list[0]['type_of_trial'] == "SJ":
        #vertical_take_off_velocity = st.number_input("Give the time of vertical take off velocity")
        jump_depending_take_off_velocity = (df.loc[take_off_time, 'Velocity'] ** 2) / (2 * 9.81)
        jump_depending_time_in_air = (1 / 2) * 9.81 * (((landing_time - take_off_time) / 1000 ) / 2 ) ** 2 
    
    # Find the Jump depending on time in Air for DJ Trial:
    if url_list[0]['type_of_trial'] == "DJ":
        jump_depending_take_off_velocity = (df.loc[take_off_time, 'Velocity'] ** 2) / (2 * 9.81)
        jump_depending_time_in_air = (1 / 2) * 9.81 * (((landing_time - take_off_time) / 1000 ) / 2 ) ** 2 
        rsi = jump_depending_time_in_air / ((take_off_time - start_try_time) / 1000 )
    
    

    # --- Initialising SessionState ---
    if "load_state" not in st.session_state:
        st.session_state.load_state = False

    ######### ###### ######### ######## BRUSHED AREA ########### ########## ###########
    if brushed_submitted or st.session_state.load_state:
        st.session_state.load_state = True

        df_brushed = df[(df.index >= user_time_input_min_jumps_table) & (df.index <= user_time_input_max_jumps_table)]

        ######### ######## ########## FIND JUMP DEPENDING ON IMPLUSE FOR CMJ, SJ ############ ########### ############
        if url_list[0]['type_of_trial'] == "CMJ" or url_list[0]['type_of_trial'] == "SJ" :
            #Find the Impluse GRF:
            df_brushed['Impulse_grf'] = df_brushed['Force'] * (1/1000)
            impulse_grf = df_brushed['Impulse_grf'].sum()
            #Find the Impulse BW:
            impulse_bw_duration = (user_time_input_max_jumps_table - user_time_input_min_jumps_table) / 1000
            impulse_bw = url_list[0]['weight'] * 9.81 * impulse_bw_duration
            # Find the Velocity depeding on Impulse:
            velocity_momentum1 = (impulse_grf - impulse_bw) / url_list[0]['weight']
            # Find the Jump:
            jump_depending_impluse = (velocity_momentum1 ** 2) / (9.81 * 2)
            rsi_duration = (take_off_time - start_try_time) / 1000
            rsi = jump_depending_impluse / rsi_duration
            
        ##### #### #### ##### FIND THE RFD linear igression ##### #### #### #### #####
        l_rfd1=[] 
        # l_emg1=[] # l_emg2=[] # l_emg3=[]
        b_rfd1=[]
        #b_rfd1=[]
        # l_emg1=[] # l_emg2=[] # l_emg3=[] # b_emg1=[] # b_emg2=[] # b_emg3=[]
        headers_list_rfd1=[]
        # headers_list_emg1=[] # headers_list_emg2=[] # headers_list_emg3=[]
        rfd_df1=pd.DataFrame()

        if url_list[0]['type_of_trial'] == "ISO":
            # The whole RFD:
            X_all = df.loc[from_time_rfd_iso:till_time_rfd_iso,'Rows_Count'] - df.loc[from_time_rfd_iso:till_time_rfd_iso,'Rows_Count'].mean()
            Y_all = df.loc[from_time_rfd_iso:till_time_rfd_iso,'Force'] - df.loc[from_time_rfd_iso:till_time_rfd_iso,'Force'].mean()
            b_rfd1_whole = (X_all*Y_all).sum() / (X_all ** 2).sum()
            RFP_Total = pd.Series(b_rfd1_whole)
        else:
            # The whole RFD:
            X_all = df_brushed['Rows_Count'] - df_brushed['Rows_Count'].mean()
            Y_all = df_brushed['Force'] - df_brushed['Force'].mean()
            b_rfd1_whole = (X_all*Y_all).sum() / (X_all ** 2).sum()
            RFP_Total = pd.Series(b_rfd1_whole)

        k=0
        # IF TRIAL IS ISO
        if url_list[0]['type_of_trial'] == "ISO":
            for i in range(int(from_time_rfd_iso),int(till_time_rfd_iso+1),50):  
                ###### FIND RFD on selected time period ######                
                X = df.loc[from_time_rfd_iso:i:1,'Rows_Count'] - df.loc[from_time_rfd_iso:i:1,'Rows_Count'].mean()
                Y = df.loc[from_time_rfd_iso:i:1,'Force'] - df.loc[from_time_rfd_iso:i:1,'Force'].mean()
                b_rfd1 = (X*Y).sum() / (X ** 2).sum()
                headers_list_rfd1.append("RFD 0 - "+(str(k)))
                k += 50
                l_rfd1.append(b_rfd1)

            # Create the final dataframe for RFD 
            if rfd_df1.empty:
                rfd_df1 = pd.DataFrame([l_rfd1])
                cols = len(rfd_df1.axes[1])
                rfd_df1.columns = [*headers_list_rfd1]
            else:
                to_append = l_rfd1
                rfd_df1_length = len(rfd_df1)
                rfd_df1.loc[rfd_df1_length] = to_append

        # IF TRIAL IS NOT ISO:
        # emg_df1=pd.DataFrame() # emg_df2=pd.DataFrame() # emg_df3=pd.DataFrame()
        else:
            for i in range(int(user_time_input_min_jumps_table),int(user_time_input_max_jumps_table+1),50):  
                ###### FIND RFD on selected time period ######
                X = df_brushed.loc[user_time_input_min_jumps_table:i:1,'Rows_Count'] - df_brushed.loc[user_time_input_min_jumps_table:i:1,'Rows_Count'].mean()
                Y = df_brushed.loc[user_time_input_min_jumps_table:i:1,'Force'] - df_brushed.loc[user_time_input_min_jumps_table:i:1,'Force'].mean()
                b_rfd1 = (X*Y).sum() / (X ** 2).sum()
                headers_list_rfd1.append("RFD 0 - "+(str(k)))
                k += 50
                l_rfd1.append(b_rfd1)
                
                #FIND R-EMG
                # X = df_brushed.loc[user_time_input_min_jumps_table:i:1,'Rows_Count'] - df_brushed.loc[user_time_input_min_jumps_table:i:1,'Rows_Count'].mean()
                # Y1 = df_brushed.loc[user_time_input_min_jumps_table:i:1,'pre_pro_signal_EMG_1'] - df_brushed.loc[user_time_input_min_jumps_table:i:1,'pre_pro_signal_EMG_1'].mean()
                # Y2 = df_brushed.loc[user_time_input_min_jumps_table:i:1,'pre_pro_signal_EMG_2'] - df_brushed.loc[user_time_input_min_jumps_table:i:1,'pre_pro_signal_EMG_2'].mean()
                # Y3 = df_brushed.loc[user_time_input_min_jumps_table:i:1,'pre_pro_signal_EMG_3'] - df_brushed.loc[user_time_input_min_jumps_table:i:1,'pre_pro_signal_EMG_3'].mean()
                # b_emg1 = (X*Y1).sum() / (X ** 2).sum()
                # b_emg2 = (X*Y2).sum() / (X ** 2).sum()
                # b_emg3 = (X*Y3).sum() / (X ** 2).sum()
                # headers_list_emg1.append("EMG_1-"+str(i))
                # headers_list_emg2.append("EMG_2-"+str(i))
                # headers_list_emg3.append("EMG_3-"+str(i))
                # l_emg1.append(b_emg1)
                # l_emg2.append(b_emg2)
                # l_emg3.append(b_emg3)

            # Create the final dataframe for RFD 
            if rfd_df1.empty:
                rfd_df1 = pd.DataFrame([l_rfd1])
                cols = len(rfd_df1.axes[1])
                rfd_df1.columns = [*headers_list_rfd1]
            else:
                to_append = l_rfd1
                rfd_df1_length = len(rfd_df1)
                rfd_df1.loc[rfd_df1_length] = to_append

        # #Dataframe for EMG1
        # if emg_df1.empty:
        #     emg_df1 = pd.DataFrame([l_emg1])
        #     cols = len(emg_df1.axes[1])
        #     emg_df1.columns = [*headers_list_emg1]
        # else:
        #     to_append = emg_df1
        #     emg_df1_length = len(emg_df1)
        #     emg_df1.loc[emg_df1_length] = to_append
        
        # #Dataframe for EMG2
        # if emg_df2.empty:
        #     emg_df2 = pd.DataFrame([l_emg2])
        #     cols = len(emg_df2.axes[1])
        #     emg_df2.columns = [*headers_list_emg2]
        # else:
        #     to_append = emg_df2
        #     emg_df2_length = len(emg_df2)
        #     emg_df2.loc[emg_df2_length] = to_append

        # #Dataframe for EMG3
        # if emg_df3.empty:
        #     emg_df3 = pd.DataFrame([l_emg3])
        #     cols = len(emg_df3.axes[1])
        #     emg_df3.columns = [*headers_list_emg3]
        # else:
        #     to_append = emg_df3
        #     emg_df3_length = len(emg_df3)
        #     emg_df3.loc[emg_df3_length] = to_append


        ############ ########### DIPLAY SPECIFIC CALCULATIONS ON BRUSHED AREA CMS , SJ , DJ ######## ######### ############
        rms_1_normalized = float("nan")
        rms_2_normalized = float("nan")
        rms_3_normalized = float("nan")

        with st.expander('Show Specific Calculations' , expanded=True):
            st.write('Time Period : from', user_time_input_min_jumps_table, "to ", user_time_input_max_jumps_table)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                    if url_list[0]['type_of_trial'] == "ISO":
                        st.write('Force-Mean:', round(df_brushed["Force"].mean(),4))
                        st.write('Force-Min:', round(min(df_brushed['Force']),4))
                        st.write('Force-Max:', round(max(df_brushed['Force']),4))
                    else:
                        st.write('Force-Mean:', round(df_brushed["Force"].mean(),4))
                        st.write('Force-Min:', round(min(df_brushed['Force']),4))
                        st.write('Force-Max:', round(max(df_brushed['Force']),4))

            with col3:
                    if url_list[0]['type_of_trial'] == "ISO":
                        st.write('RMS_1_ISO-Mean:', round(df_brushed["RMS_1"].mean(),4))
                        st.write('RMS_2_ISO-Mean:', round(df_brushed['RMS_2'].mean(),4))
                        st.write('RMS_3_ISO-Mean:', round(df_brushed['RMS_3'].mean(),4))
                    else:
                        st.write('RMS_1-Mean:', round(df_brushed["RMS_1"].mean(),4))
                        st.write('RMS_2-Mean:', round(df_brushed['RMS_2'].mean(),4))
                        st.write('RMS_3-Mean:', round(df_brushed['RMS_3'].mean(),4))
            #if url_list[0]['type_of_trial'] == "CMJ" or url_list[0]['type_of_trial'] == "DJ" or url_list[0]['type_of_trial'] == "SJ":
            with col4:
                    if url_list[0]['type_of_trial'] != "ISO" and rms_1_iso:
                        rms_1_normalized = df_brushed["RMS_1"].mean() / rms_1_iso
                        st.write("RMS 1 Norm:", round(rms_1_normalized,4))
                    if url_list[0]['type_of_trial'] != "ISO" and rms_2_iso:
                        rms_2_normalized = df_brushed["RMS_2"].mean() / rms_2_iso
                        st.write("RMS 2 Norm:", round(rms_2_normalized,4))
                    if url_list[0]['type_of_trial'] != "ISO" and rms_3_iso:
                        rms_3_normalized = df_brushed["RMS_3"].mean() / rms_3_iso
                        st.write("RMS 3 Norm:", round(rms_3_normalized,4))          
            if url_list[0]['type_of_trial'] == "CMJ" or url_list[0]['type_of_trial'] == "SJ":
                with col2:
                        st.write('Jump (Impluse):', round(jump_depending_impluse,4))
                        st.write('Jump (Take Off Velocity):', round(jump_depending_take_off_velocity,4))
                        st.write('Jump (Time in Air):', round(jump_depending_time_in_air,4))
                        st.write("RSI-mod", round(rsi,4))
            if url_list[0]['type_of_trial'] == "DJ":
                with col2:
                        st.write('Jump (Take Off Velocity):', round(jump_depending_take_off_velocity,4))
                        st.write('Jump (Time in Air):', round(jump_depending_time_in_air,4))
                        st.write('RSI:', round(rsi,4))
                        
        #Display Dataframe in Datatable
        if url_list[0]['type_of_trial'] != "ISO":
            with st.expander("Show Data Table", expanded=True):
                selected_filtered_columns = st.multiselect(
                label='What column do you want to display', default=('Time', 'Force', 'Acceleration', 'Velocity', 'RMS_1', 'RMS_2','RMS_3'), help='Click to select', options=df_brushed.columns)
                st.write(df_brushed[selected_filtered_columns])
                #Button to export results
                st.download_button(
                    label="Export table dataset",
                    data=df_brushed[selected_filtered_columns].to_csv(),
                    file_name=url_list[0]['filename'] +'.csv',
                    mime='text/csv',
                )

        ###### FINAL RESULTS #####
        st.write("---")
        st.write('**Final Results Table for user : {}**'.format(url_list[0]['fullname']))
        specific_metrics = [""]
        if url_list[0]['type_of_trial'] == 'ISO':
            specific_metrics = {#'Unit': ['results'],
                    'Fullname' : url_list[0]['fullname'],
                    'Occupy' : url_list[0]['occupy'],
                    'Type of try' : url_list[0]['type_of_trial'],
                    'Filename' : url_list[0]['filename'],
                    'Body Mass (kg)': url_list[0]['weight'],
                    'Jump (m/s)' : [jump_depending_impluse],
                    'RSI m/s' : [rsi],
                    'RMS 1 Mean' : [df_brushed['RMS_1'].mean()],
                    'RMS 1 Norm' : [rms_1_normalized],
                    'RMS 2 Mean' : [df_brushed['RMS_2'].mean()],
                    'RMS 2 Norm' : [rms_2_normalized] if rms_2_normalized is not None else {},
                    'RMS 3 Mean' : [df_brushed['RMS_3'].mean()],
                    'RMS 3 Norm' : [rms_3_normalized] if rms_3_normalized is not None else {},
                    'Force Mean (N)' : [df_brushed['Force'].mean()],
                    'Force Max (N)' : [max(df_brushed['Force'])],  
                    'RFD Total ' + str(from_time_rfd_iso) + ' - ' + str(till_time_rfd_iso) : [b_rfd1_whole]                
                    }
        else:
            specific_metrics = {#'Unit': ['results'],
                    'Fullname' : url_list[0]['fullname'],
                    'Occupy' : url_list[0]['occupy'],
                    'Type of try' : url_list[0]['type_of_trial'],
                    'Filename' : url_list[0]['filename'],
                    'Body Mass (kg)': url_list[0]['weight'],
                    'Jump (m/s)' : [jump_depending_impluse],
                    'RSI m/s' : [rsi],
                    'RMS 1 Mean' : [df_brushed['RMS_1'].mean()],
                    'RMS 1 Norm' : [rms_1_normalized],
                    'RMS 2 Mean' : [df_brushed['RMS_2'].mean()],
                    'RMS 2 Norm' : [rms_2_normalized] if rms_2_normalized is not None else {},
                    'RMS 3 Mean' : [df_brushed['RMS_3'].mean()],
                    'RMS 3 Norm' : [rms_3_normalized] if rms_3_normalized is not None else {},
                    'Force Mean (N)' : [df_brushed['Force'].mean()],
                    'Force Max (N)' : [max(df_brushed['Force'])],  
                    'RFD Total ' + str(user_time_input_min_jumps_table) + ' - ' + str(user_time_input_max_jumps_table) : [b_rfd1_whole]                
                    }

        specific_metrics_df = pd.DataFrame(specific_metrics)
        #Combine all dataframes to one , for the final export
        final_results_df = pd.concat([specific_metrics_df, rfd_df1], axis=1, join='inner')
        final_results_df =np.round(final_results_df, decimals = 4)        

        st.dataframe(final_results_df.T, use_container_width=True )
        #st.write(specific_metrics)
        def to_excel(final_results_df):
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            final_results_df.to_excel(writer, index=False, sheet_name='Sheet1')
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            format1 = workbook.add_format({'num_format': '0.00'}) 
            worksheet.set_column('A:A', None, format1)  
            writer.save()
            processed_data = output.getvalue()
            return processed_data

        df_xlsx = to_excel(final_results_df)
        st.download_button(label='Export Final Results xlsx',
                                        data=df_xlsx ,
                                        file_name= url_list[0]['filename'] + '_.xlsx')

        st.download_button(
            label="Export Final Results",
            data=final_results_df.to_csv(),
            file_name=url_list[0]['filename'] +'_final_results.csv',
            mime='text/csv',
                )

        with st.form("Insert results to Database:"):   
            verify_check_box_insert_final_results = st.text_input( "Please type Verify to insert the final results to database")

            submitted_button_insert_final_results = st.form_submit_button("Insert Results")
        
        if submitted_button_insert_final_results:
            # check if this id allready exists in database
            def check_if_this_id_entry_exists(supabase):
                query=con.table("jumps_statistics_table").select("id").eq("id", url_id_number_input).execute()
                return query
            query = check_if_this_id_entry_exists(con)
            
            # Check if list query.data is empty or not
            if query.data:
                st.warning("This entry with this id allready exists in table")
            else:
                st.write("Mpika Else", url_id_number_input)
                # After Export , try to insert these values to statistics table      
                def add_entries_to_jumps_statistics_table(supabase):
                        value = {'id': url_id_number_input, 'fullname': url_list[0]['fullname'], "age": url_list[0]['age'] , "height": url_list[0]['height'], "weight": url_list[0]['weight'], 'type_of_trial': url_list[0]['type_of_trial'], 'filename': url_list[0]['filename'], "filepath": url_list[0]['filepath'], 'occupy': url_list[0]['occupy'], 
                                'jump': round(jump_depending_impluse,4), 'rms_1_mean': df_brushed['RMS_1'].mean(), 'rms_2_mean': df_brushed['RMS_2'].mean(), 'rms_3_mean': df_brushed['RMS_3'].mean(),  'force_mean': round(df_brushed['Force'].mean(),4), 
                                'force_max': round(max(df_brushed['Force']),4), 'rms_1_norm': rms_1_normalized, 'rms_2_norm': rms_2_normalized, 'rms_3_norm': rms_3_normalized }
                        data = supabase.table('jumps_statistics_table').insert(value).execute()
                def main():
                    new_entry = add_entries_to_jumps_statistics_table(con)
                main()
                st.success('Thank you! A new entry has been inserted to database!')

                def select_all_from_jumps_statistics_table():
                    query=con.table("jumps_statistics_table").select("*").execute()
                    return query
                query = select_all_from_jumps_statistics_table()
                df_jumps_statistics_table = pd.DataFrame(query.data)
                st.write("The datatable with Final Results:", df_jumps_statistics_table)



    ##################### ################### UN BRUSHED AREA ##################### ######################## ###################
    else:
        with st.expander("Show Specific Calculations", expanded=True):
            st.caption("Whole Time Period")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                    st.write('Force-Mean:', round(df["Force"].mean(),4))
                    st.write('Force-Min:', round(min(df['Force']),4))
                    st.write('Force-Max:', round(max(df['Force']),4))
            with col2:
                    st.write('RMS_1-Mean:', round(df["RMS_1"].mean(),4))
                    st.write('RMS_2-Mean:', round(df["RMS_2"].mean(),4))
                    st.write('RMS_3-Mean:', round(df["RMS_3"].mean(),4))   
            with col3:
                    st.write("")
            with col4:
                    st.write("")       
        
        #Display Dataframe in Datatable
        with st.expander("Show Data Table", expanded=True):
            selected_clear_columns = st.multiselect(
            label='What column do you want to display', default=('Time', 'Force', 'Acceleration', 'Velocity', 'RMS_1', 'RMS_2', 'RMS_3'), help='Click to select', options=df.columns)
            st.write(df[selected_clear_columns])
            #Button to export results
            st.download_button(
                label="Export table dataset",
                data=df[selected_clear_columns].to_csv(),
                file_name=url_list[0]['filename'] + '.csv',
                mime='text/csv',
            )
    #Values Sidebar
    
    with st.sidebar.expander(("Information about the Trial"), expanded=True):
        st.write('**Name**:', url_list[0]['fullname'])
        st.write('**Age**:', url_list[0]['age'])
        st.write('**Height**:', url_list[0]['height'])
        st.write('**Body mass is**:', round(url_list[0]['weight'],4), 'kg')
        st.write('**Type of try**:', url_list[0]['type_of_trial'])
        st.write('**File Name**:', url_list[0]['filename'])
        st.write('**Occupy:**', url_list[0]['occupy'])
        st.write('**Jump:**', jump_depending_impluse)
        if url_list[0]['type_of_trial'] == "CMJ":
            st.write('**Start Trial starts at**:', start_try_time, 'ms')
            st.write('**Take Off Time starts at**:', take_off_time, 'ms')
            st.write('**Landing Time at**:', landing_time, 'ms')
        
            
