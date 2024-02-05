import streamlit as st
import os
import pandas as pd 

def list_files_in_directory():
    cwd = '.'
    #st.write("##### Your current working directory is:", cwd)
    # prints parent directory
    # pwd = os.path.abspath(os.path.join(cwd, os.pardir))
    # st.write('##### Your parent working directory is:', pwd)
    #pwd = os.listdir(path)
    cwd_items_list = os.listdir(cwd)
    #st.write("##### This current working directory has the following list of itmes:", cwd_items_list)

    select_item_of_cwd_list = st.sidebar.selectbox(label='Select an item of the current working directory:', options=cwd_items_list)
    #st.write('You select this item of current working directory:', select_item_of_cwd_list)
    isFolder = os.path.isdir(cwd+'/'+select_item_of_cwd_list)

    if isFolder == True:
        #st.write("##### You select a folder with name:", select_item_of_cwd_list)
        selected_folder_items_list = os.listdir(select_item_of_cwd_list)
        #st.write("##### The folder",select_item_of_cwd_list,"contains the following files:", selected_folder_items_list)
        #for item in select_item_of_cwd_list:
        
        select_item_of_selected_folder_list = st.sidebar.selectbox(label='Select an file of the selected folder:', options=selected_folder_items_list)
        st.write("**File path:**", select_item_of_cwd_list,"/",select_item_of_selected_folder_list )
        if st.sidebar.checkbox(label="Check the content of the csv file:"):
            st.sidebar.write("The Folder size is: " ,os.path.getsize(select_item_of_cwd_list),"bytes")
            st.sidebar.write("The file size is: " , round((os.path.getsize(select_item_of_cwd_list + '/' + select_item_of_selected_folder_list) * (0.0009765625 * 0.0009765625)),2),"MB")
            st.sidebar.write("Do you want to delete this file?")
            with st.sidebar.form("Delete this file"):
                    input_verify = st.text_input("Type Yes to delete")
                    delete_button = st.form_submit_button("Delete")
                    if delete_button:
                        os.remove(select_item_of_cwd_list + '/' + select_item_of_selected_folder_list)

            if select_item_of_selected_folder_list.endswith('.csv'):
                st.write("The csv file containts the below content:")
                # url = "https://sportsmetrics.geth.gr/storage/1 CJ 2ND 0007800f8127_emg 00078065e001_2022-03-25_16-45-21 (1).csv"
                # st.write("check out this [link](%s)" % url)
                # st.markdown("check out this [link](%s)" % url)
                df_preview = pd.read_csv(select_item_of_cwd_list + '/' + select_item_of_selected_folder_list)
                st.dataframe(df_preview, use_container_width=True)
    else:
        st.write("You select a file with name:", select_item_of_cwd_list)


    #cwd+'/'+select_item_of_cwd_list

# # Checks if path is a directory
# isDirectory = os.path.isdir(fpath)

    
    # file_info_list = []
    # for filename in files_list:
    #     file_path = os.path.join(path, filename)
    #     st.write("file_path",file_path)
    #     #st.write("walk",os.walk(path))
    #     # check if the item is file:
    #     if os.path.isfile(file_path):
    #         file_info = {
    #             "name": filename,
    #             # get the file size in bytes:
    #             "size": os.path.getsize(file_path)
    #         }
    #         file_info_list.append(file_info)
    # return file_info_list

file_list = list_files_in_directory()
# df = pd.DataFrame(file_list)
# st.write(df)
# path = '.'
# st.write('a',os.chdir(path), path)

# for p,n,f in os.walk(os.getcwd()):
#     for a in f:
#         a = str(a)
#         if a.endswith('.csv'):
#             st.write(a)
#             st.write(p)


