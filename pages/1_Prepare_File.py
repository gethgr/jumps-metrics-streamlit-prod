import os
import streamlit as st
import pandas as pd
import ftplib
import tempfile
from pathlib import Path
 
############## ############## PAGE 1 PREPARE THE FILE ############# ############# ############## ##############
      
      
st.set_page_config(
    page_title="Tefaa Metrics",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
   
)

st.title('Prepare the file')
st.write('**Into the below form insert the txt file of unconverted raw values from the force platform.**')

st.sidebar.info("Instructions")
st.sidebar.info("-Use the adjacent form and choose the txt file with the raw data from the force platform for the trial that interests you.")
st.sidebar.info("-Give the value of the platform mass.")
st.sidebar.info("-Use the slider to select the time period you want.")
st.sidebar.info("-Finaly check the Verify box and export the file from the 'Export File' button.")



with st.sidebar.form("File", clear_on_submit=False):

    with st.expander("Show File", expanded=True):
        filepath = st.file_uploader("Choose a txt file")
    submitted_file = st.form_submit_button("Submit file")

if submitted_file:
    if filepath:
        filename_with_extension = filepath.name
        filename = os.path.splitext(filename_with_extension)[0]

        def storage_connection():
            hostname = st.secrets["hostname"]
            username = st.secrets["username"]
            password = st.secrets["password"]
            
            return hostname,username,password
        hostname,username,password = storage_connection()
        ftp = ftplib.FTP(hostname,username,password)
        
        # This is the method to take the temporary path of the uploaded file and the value in bytes of it.
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            fp_PosixPath = Path(tmp_file.name)
            fp_PosixPath.write_bytes(filepath.getvalue())
        # This is to take the str of PosixPath.
        fp_str = str(fp_PosixPath)
        # This is our localfile's path in str.
        localfile = fp_str
        # This is the remote path of the server to be stored.
        
        remotefile='/sportsmetrics.geth.gr/storage/' + filename_with_extension
        #remotefile='/home/ftp_user/' + filename_with_extension

        # This is the method to store the localfile in remote server through ftp.
        with open(localfile, "rb") as file:
            ftp.storbinary('STOR %s' % remotefile, file)
        ftp.quit()
        
        #filepath='/home/ftp_user/' + filename_with_extension
        
        filepath="https://sportsmetrics.geth.gr/storage/" + filename_with_extension
        st.sidebar.write("thank you")  


with st.expander("Show File Form", expanded=True):
    uploaded_file = st.file_uploader("Choose a file")
platform_mass = st.number_input("Give the platfrom mass:")

#@st.cache(allow_output_mutation=True)
def get_data():
    if uploaded_file:
        df_raw_data = pd.read_csv(uploaded_file,  sep='\s+', skiprows=10, index_col = None) # 
        #Define Header columns
        columns_count = len(df_raw_data.axes[1])
        # if columns_count == 4:
        #     df_raw_data.columns = ['Time', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4']
        #st.write(columns_count)
        if columns_count == 5:
            df_raw_data.columns = ['Time', 'Col_2', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4']
        if columns_count == 6:
            df_raw_data.columns = ['Time', 'Col_2', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4']
        if columns_count == 8:
            df_raw_data.columns = ['Time', 'Col_2', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_7', 'Col_8']
        if columns_count == 9:
            df_raw_data.columns = ['Time', 'Col_2', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_7', 'Col_8', 'Col_9']
        if columns_count == 10:
            df_raw_data.columns = ['Time', 'Col_2', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_7', 'Col_8', 'Col_9', 'Col_10']
        if columns_count == 11:
            df_raw_data.columns = ['Time', 'Col_2', 'Mass_1', 'Mass_2', 'Mass_3', 'Mass_4', 'Col_7', 'Col_8', 'Col_9', 'Col_10','Col_11']
        C = 406.831
        #sr = 1000
        resolution = 16
        # Calculate for A Sensor Mass $ Weight
        Vfs_1 = 2.00016
        df_raw_data['Mass_1'] = df_raw_data['Mass_1'] * C / (Vfs_1 * ( (2**resolution) - 1 ) )
        # Calculate for B Sensor Mass $ Weight
        Vfs_2 = 2.00002
        df_raw_data['Mass_2'] = df_raw_data['Mass_2'] * C / (Vfs_2 * ( (2**resolution) - 1 ) )
        # Calculate for C Sensor Mass $ Weight
        Vfs_3 = 2.00057
        df_raw_data['Mass_3'] = df_raw_data['Mass_3'] * C / (Vfs_3 * ( (2**resolution) - 1 ) )
        # Calculate for D Sensor Mass $ Weight
        Vfs_4 = 2.00024
        df_raw_data['Mass_4'] = df_raw_data['Mass_4'] * C / (Vfs_4 * ( (2**resolution) - 1 ) )
        # Calculate the sum of all sensors Mass $ Weight
        df_raw_data['Mass_Sum'] = (df_raw_data['Mass_1'] + df_raw_data['Mass_2'] + df_raw_data['Mass_3'] + df_raw_data['Mass_4']) - platform_mass
        df_raw_data['Rows_Count'] = df_raw_data.index
        
        return df_raw_data

if uploaded_file:
    
    df_raw_data= get_data()
    #st.write(columns_count)
    
    # if st.button('Reload Dataframe with Raw Data'):
    #     get_data()
    if df_raw_data is not None:
        min_time = int(df_raw_data.index.min())
        max_time = int(df_raw_data.index.max())
        selected_time_range = st.slider('Select the whole time range of the graph, per 100', min_time, max_time, (min_time, max_time), 1)
        selected_area = (df_raw_data.Rows_Count.between(selected_time_range[0], selected_time_range[1]) )
        df_prepared = pd.DataFrame(df_raw_data[selected_area])
        
        st.line_chart(df_prepared['Mass_Sum'])
        # To Drop the unnecessary Columns
        df_prepared.drop(['Rows_Count'], axis = 1, inplace=True)
        filename = uploaded_file.name
        # To Get only the filename without extension (.txt)
        final_filename = os.path.splitext(filename)[0]
        st.write("The file name of your file is : ", final_filename)
        show_df_prepared = st.checkbox("Display the final dataframe")
        if show_df_prepared:
            st.dataframe(df_prepared)
        # if platform_mass >1:
        #     st.download_button(
        #         label="Export File",
        #         data=df_prepared.to_csv(index=False),
        #         file_name=final_filename +'.csv',
        #         mime='text/csv',
        #     )


        export = st.checkbox('Verify you have insert proper Platform Mass Value:')

        if export:
            if 0 <= platform_mass <= 10:
                st.success("You are able to export your data.")
                st.download_button(
                    label="Export File",
                    data=df_prepared.to_csv(index=False),
                    file_name=final_filename +'.csv',
                    mime='text/csv',
                )
            else:
                st.warning("Please give correct platform mass!")



