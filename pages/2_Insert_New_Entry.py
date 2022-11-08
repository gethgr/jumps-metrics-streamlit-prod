
import os
import streamlit as st
import pandas as pd
from supabase import create_client, Client
import ftplib
import tempfile
from pathlib import Path
import numpy as np


############# ############## PAGE 2 INSERT TO DATABASE USER+TRIAL ############## ############ #############################
st.set_page_config(
    page_title="Tefaa Metrics",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    
)

#Make the connection with Supabase - Database:
@st.experimental_singleton
def init_connection():
    url = st.secrets["supabase_url"]
    key = st.secrets["supabase_key"]
    #client = create_client(url, key)
    return create_client(url, key)
con = init_connection()


st.sidebar.info("Try to insert a new entry to database.")
st.sidebar.info("Fill in all the necessary fields of the form.")
st.sidebar.info("After that, click on the submit values to import the entry into database.")
st.sidebar.info("Finally go to Calculate Results to see the metrics.")


st.title("Import Entry to Database!")

# filepath1 = st.file_uploader("Choose a file1")
# #filepath2 =os.path.basename(fileitem.filepath1)
# #fil = pathlib.Path(filepath1.name)
# filepath1.name

def select_all_from_jumps_table():
    query=con.table("jumps_table").select("*").execute()
    return query
query = select_all_from_jumps_table()


df_jumps_table = pd.DataFrame(query.data)
df_jumps_table_unique_values = df_jumps_table.drop_duplicates(subset = ["fullname"])

df_jumps_table_unique_values = df_jumps_table_unique_values.shift()
df_jumps_table_unique_values.loc[0] = [int, float("Nan"), '-', '-', '-', '-','-','-', float("Nan"), float("Nan"), 0]
fullname_input = st.selectbox("Select a person from the database or fill in the fields below. " , (df_jumps_table_unique_values['fullname']))
row_index = df_jumps_table_unique_values.index[df_jumps_table_unique_values['fullname']==fullname_input].tolist()
st.markdown("""---""")

#Create the Form to submit data to database:
with st.form("Create a new entry", clear_on_submit=False):
    col1,col2=st.columns(2)
    with col1:
        fullname = st.text_input("Fullname", value = df_jumps_table_unique_values.loc[row_index[0]]['fullname'])
        age = st.number_input("Age", value = int(df_jumps_table_unique_values.loc[row_index[0]]['age']), min_value=0, max_value=100, step=1)
        height = st.number_input("Height in cm", value = df_jumps_table_unique_values.loc[row_index[0]]['height'])
        weight = st.number_input("Weight in kg", value = df_jumps_table_unique_values.loc[row_index[0]]['weight'])
    with col2:
        email = st.text_input("Email address")
        occupy = st.text_input("Occupy", value = df_jumps_table_unique_values.loc[row_index[0]]['occupy'])
        type_of_trial = st.selectbox("Kind of Trial", ('-','CMJ', 'SJ','DJ','ISO' ))
        filepath = st.file_uploader("Choose a file", type="csv")
    #checkbox_val = st.checkbox("Form checkbox")
    submitted = st.form_submit_button("Submit values")
    
    if submitted:
        
        if fullname and age and height and weight and occupy and type_of_trial !='-' and filepath:
            
            filename_with_extension = filepath.name
            # Filename without extension
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

            # This is the method to store the localfile in remote server through ftp.
            with open(localfile, "rb") as file:
                ftp.storbinary('STOR %s' % remotefile, file)
            ftp.quit()
            
            filepath="https://sportsmetrics.geth.gr/storage/" + filename_with_extension
                     
            list = (fullname,email,occupy,type_of_trial,filename)
            def add_entries_to_jumps_table(supabase):
                value = {'fullname': fullname, 'email': email, 'occupy': occupy, 'type_of_trial': type_of_trial,
                        'filename': filename, "filepath": filepath, "height": height, "weight": weight, "age": age }
                data = supabase.table('jumps_table').insert(value).execute()
            def main():
                new_entry = add_entries_to_jumps_table(con)
            main()
            st.success('Thank you! A new entry has been inserted to database!')
            st.write(list)
        else:
            st.error("One of the field values is missing")
#@st.experimental_memo(ttl=600)
def select_all_from_jumps_table():
    query=con.table("jumps_table").select("*").execute()
    return query
jumps_table_all = select_all_from_jumps_table()
df_all_from_jumps_table = pd.DataFrame(jumps_table_all.data)


# url = st.text_input("Paste the desire url")
#
# if url:
#     storage_options = {'User-Agent': 'Mozilla/5.0'}
#     df = pd.read_csv(url,storage_options=storage_options)
#     st.write(df)






