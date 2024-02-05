import os
import streamlit as st
import pandas as pd
from supabase import create_client, Client
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.ensemble import RandomForestRegressor


       
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
df['Duration I'] = df['From time'] - df['Start trial time']
df['Duration II'] = df['Till time'] - df['Start trial time']
df['Duration III'] = df['Landing time'] - df['Start trial time']
df['Duration IV'] = df['Till time'] - df['From time']
df['velocity_momentum'] = (df['Impulse gravity'] - df['Impulse body weight']) / df['Weight']


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
        occupy_search = st.text_input("Ιδιότητα:")
    with col1:
        unique_fullnames = df['Fullname'].unique()
        options =[" "] + [unique_fullnames[i] for i in range (0, len(unique_fullnames)) ]
        fullname_search = st.selectbox("Επέλεξε χρήστη:" , options = options)
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

st.download_button(
    label="Export table dataset",
    data=df[df['Fullname']== fullname_search][df['Type of trial']== type_of_trial_search][df['Occupy'] == occupy_search].to_csv(),
    file_name='export.csv',
    mime='text/csv',
)

# # add tabs
# tab1, tab2, tab3 = st.tabs(["Data Info", "Numeric Features", "Categorical Features"])
# with tab1:
# #   if uploaded_data is not None:
# #     # extract meta-data from the uploaded dataset
    

#     row_count = df.shape[0]

#     column_count = df.shape[1]
        
#     # Use the duplicated() function to identify duplicate rows
#     duplicates = df[df.duplicated()]
#     duplicate_row_count =  duplicates.shape[0]

#     missing_value_row_count = df[df.isna().any(axis=1)].shape[0]

#     table_markdown = f"""
#         | Description | Value | 
#         |---|---|
#         | Number of Rows | {row_count} |
#         | Number of Columns | {column_count} |
#         | Number of Duplicated Rows | {duplicate_row_count} |
#         | Number of Rows with Missing Values | {missing_value_row_count} |
#         """


    

#     # get feature names
#     columns = list(df.columns)

#     # create dataframe
#     column_info_table = pd.DataFrame({
#         "column": columns,
#         "data_type": df.dtypes.tolist()
#     })
#     col1 , col2 , col3 = st.columns(3)
#     with col1:
#         st.header("Meta-data")
#         st.markdown(table_markdown)
#     with col2:
#         st.header("Columns Type")
#         # display pandas dataframe as a table
#         st.dataframe(column_info_table, hide_index=True)
#     with col3:
#         st.write("##### List of unique persons who made trial: ")
#         st.write(df.Fullname.unique())
# with tab2:
#     # find numeric features  in the dataframe
#     numeric_cols = df.select_dtypes(include='number').columns.tolist()

#     # add selection-box widget
#     selected_num_col = st.selectbox("Which numeric column do you want to explore?", numeric_cols)

#     col1, col2 = st.columns(2, gap='small')
#     with col1:
        
#         st.header(f"{selected_num_col} - Statistics")
        
#         col_info = {}
#         col_info["Number of Unique Values"] = len(df[selected_num_col].unique())
#         col_info["Number of Rows with Missing Values"] = df[selected_num_col].isnull().sum()
#         col_info["Number of Rows with 0"] = df[selected_num_col].eq(0).sum()
#         col_info["Number of Rows with Negative Values"] = df[selected_num_col].lt(0).sum()
#         col_info["Average Value"] = df[selected_num_col].mean()
#         col_info["Standard Deviation Value"] = df[selected_num_col].std()
#         col_info["Minimum Value"] = df[selected_num_col].min()
#         col_info["Maximum Value"] = df[selected_num_col].max()
#         col_info["Median Value"] = df[selected_num_col].median()

#         info_df = pd.DataFrame(list(col_info.items()), columns=['Description', 'Value'])

#         # display dataframe as a markdown table
#         st.dataframe(info_df)
   
#     with col2:
#         st.header("Histogram")
#         fig = px.histogram(df, x=selected_num_col)
#         st.plotly_chart(fig, use_container_width=True)

st.write("#### Display graphs")
col1,col2,col3 = st.columns(3, gap="large")
columns = [" "] + [ 'Jump (Impulse)', 'Force max', 'Duration I', 'Duration II', 'Duration III' ]

#columns = [" "] + [df.columns[i] for i in range (0, len(df.columns))]
with col1:
    fig1 = plt.figure(figsize=(8,7))
    select_person1 = st.selectbox('Select the person 1',
    options = options)
    select_column1 = st.selectbox('Select Column for person 1',
    options = columns)
    if select_column1 != " ":
        df[df["Fullname"] == select_person1][select_column1].plot.bar()
        plt.xlabel("Times of trial")
        plt.ylabel(select_column1)
        st.pyplot(fig1)

with col2:
    fig2 = plt.figure(figsize=(8,7))
    select_person2 = st.selectbox('Select the person 2',
    options = options)
    select_column2 = st.selectbox('Select Column for person 2',
    options = columns)
    if select_column2 != " ":
        df[df["Fullname"] == select_person2][select_column2].plot.bar()
        plt.xlabel("Times of trial")
        plt.ylabel(select_column2)
        st.pyplot(fig2)
    
    st.write("#")
with col3:
    fig3 = plt.figure(figsize=(8,7))
    select_person3 = st.selectbox('Select the person 3',
    options = options)
    select_column3 = st.selectbox('Select Column for person 3',
    options = columns)
    if select_column3 != " ":
        df[df["Fullname"] == select_person3][select_column3].plot.bar()
        plt.xlabel("Times of trial")
        plt.ylabel(select_column3)
        st.pyplot(fig3)


df_corr = df[['Jump (Impulse)', 'RFD Total', 'Duration I', 'Duration II', 'Duration III', 'Duration IV', 'Impulse body weight', 'Impulse gravity', 'velocity_momentum']].copy() #, 'Impulse body weight','Impulse gravity','Force max', 'Force mean',  'Force sum', 'Jump (Impulse)', 'duration_from_till_time', 'duration_from_start_trial_to_from_time']].copy()
df_corr_rfd = df[['Force max', 'Jump (Impulse)', 'RFD Total', 'Duration I', 'Duration II', 'Duration III', 'Duration IV', 'Impulse body weight', 'Impulse gravity', 'velocity_momentum']].copy() #, 'Impulse body weight','Impulse gravity','Force max', 'Force mean',  'Force sum', 'Jump (Impulse)', 'duration_from_till_time', 'duration_from_start_trial_to_from_time']].copy()
corr_matrix = df_corr.corr()
col1,col2=st.columns([2,1], gap="Large")
with col1:
    st.write("Correlation for Duration, Force max, Duration I, Duration II, Duration III / Jump (Impulse):")
    fig = plt.figure(figsize=(15, 12))
    sns.heatmap(corr_matrix, 
                
                annot=True, 
                linewidths=0.5, 
                fmt= ".2f", 
                cmap="YlGnBu")
    st.pyplot(fig)
with col2:
    st.write("Explanation of Durations:")
    st.write(" Take of time: The moment when the user does not touch the platform just before the jump. ")
    st.write(" Landing time: The moment the user steps on the platform after jumping. ")
    st.write("**Duration I**: The time moment from start trial untill the closest to zero velocity.")
    st.write("**Duration II**: The time period from start trial untill the take of time")
    st.write("**Duration III**: The time period from start trial untill the landing time ")
    st.write("Correlation table:")
    corr_matrix_jump = df_corr.corr()['Jump (Impulse)']
    corr_matrix_rfd = df_corr_rfd.corr()['RFD Total']
    st.dataframe(corr_matrix_jump, use_container_width=True)
    st.dataframe(corr_matrix_rfd, use_container_width=True)

# col1, col2 = st.columns(2, gap="large")
# with col1:
#     corr_matrix_jump = df_corr.corr()['Jump (Impulse)']
#     st.dataframe(corr_matrix_jump.sort_values(ascending=False), use_container_width=True)
# with col2:
#     st.write("")
#     #corr_matrix_rsi = df_corr.corr()['rsi']
#     #st.dataframe(corr_matrix_rsi.sort_values(ascending=False), use_container_width=True)

# ######------MODELING--------#######
# st.write("## 5. Modelling")

# # Instantiate model

# # Fit the model

# # Split data into X and y
# st.write("- Split the data into X and y :")
# X = df_corr.drop("Jump (Impulse)", axis=1)
# y = df_corr["Jump (Impulse)"]
# st.write("- Split the data into train and test sets :")
# # Split data into train and test sets
# np.random.seed(42)
# # Split into train & test set
# X_train, X_test, y_train, y_test = train_test_split(X,
#                                                     y,
#                                                     test_size=0.2)
# np.random.seed(42)
# st.write("- Create the LogisticRegression Model")
# model = RandomForestRegressor(n_jobs=20,
#                              )

# from sklearn.model_selection import RandomizedSearchCV
# # Different RandomForestRegressor hyperparameters
# rf_grid = {"n_estimators": np.arange(10, 100, 10),
#            "max_depth": [None, 3, 5, 10],
#            "min_samples_split": np.arange(2, 20, 2),
#            "min_samples_leaf": np.arange(1, 20, 2),
#            "max_features": [0.5, 1, "sqrt", "auto"],
#            "max_samples": [46]}

# # Instantiate RandomizedSearchCV model
# rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
#                                                     random_state=42),
#                               param_distributions=rf_grid,
#                               n_iter=2,
#                               cv=5,
#                               verbose=True,
#                               error_score='raise')

# # Fit the RandomizedSearchCV model
# rs_model.fit(X_train, y_train)
# st.write("** rs_model Model Score of X_test, y_test:**", rs_model.score(X_test, y_test))

# st.write("- Fit the model")

# model.fit(X_train, y_train)
# st.write("**Model Score of X_test, y_test:**", model.score(X_test, y_test))

# def jump_prediction(input_data):
#     # Changing the input data to a numpy array:
#     numpy_data= np.asarray(input_data)
#     # Reshaping the numpy:
#     input_reshaped = numpy_data.reshape(1,-1)
#     prediction = model.predict(input_reshaped)
#     st.write("Jump prediction result", prediction)
#     st.write(input_reshaped)


# st.write("Predict the jump (impulse)")

# with st.form("my_form"):
#    st.write("Inside the form")
#    #time_in_air = st.number_input("Time in air")
#    age = st.number_input("Age")
#    Height = st.number_input("Height")
#    Weight = st.number_input("Weight")
#    duration = st.number_input("Duration")
#    #Force_max = st.number_input("Force max")
#    #Force_mean = st.number_input("Force_mean")
#    #Force_min = st.number_input("Force_min")
#    #Force_sum = st.number_input("Force_sum")
#    #rsi = st.number_input("rsi")
#    #Impulse_bw_duration = st.number_input("Impulse_bw_duration")
#    #duration_from_till_time = st.number_input("duration_from_till_time")
#    #Impulse_gravity = st.number_input("Impulse_gravity")
#    #Impulse_body_weight = st.number_input("Impulse_body_weight")

#    # Every form must have a submit button.
#    submitted = st.form_submit_button("Submit")
#    if submitted:
#     st.write("Bika")
#     predict = jump_prediction([ age, Height, Weight, duration ])
    

# st.write("Outside the form")
