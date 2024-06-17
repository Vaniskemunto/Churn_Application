import streamlit as st
import pyodbc
import pandas as pd
st.set_page_config(
    page_title = 'Data Page',
    page_icon = '🏠',
    layout='wide'
)

st.title('Telecom Churn Database 📊')

# create connection and query
@st.cache_resource(show_spinner='Connecting to database...')
def init_connection():
    return pyodbc.connect(
        "DRIVER={SQL Server};SERVER="
        + st.secrets["server"]
        + ";DATABASE="
        + st.secrets["database"]
        + ";UID="
        + st.secrets["username"]
        + ";PWD="
        + st.secrets["password"]
    )

connection = init_connection()

@st.cache_data(show_spinner='Running query...')
def running_query(query):
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        st.write(cursor.description)

        df =  pd.DataFrame.from_records(rows, columns = [column[0] for column in cursor.description])
    return df
def get_all_column():
    query = "SELECT * from dbo.LP2_Telco_churn_first_3000"
    df= running_query(query)
    return df

st.write(get_all_column())
st.selectbox("Select....",options=['all columns', 'numerical columns', ' categorical columns'])

def detail():        
    col1, col2, col3 = st.columns([1,2,2])
    with col1:
        st.subheader('🎯Info:')
        st.markdown('- Dataset contains data on demographics, subscriptions and financial details of customers of a Telco.')
        st.markdown('- Dataset structured into **12 columns**.')
        st.markdown('- **3** numerical columns')            
        st.markdown('- **5** boolean features')              
        st.markdown('- **13** object features')
    with col2:
        st.markdown('''### 📚📖Dictionary:
- **Gender** - Specifies the sex of the customer
- **SeniorCitizen** - Specifies if the customer is a senior citizen or not
- **Partner** - Specifies whether the customer has a partner or not
- **Dependents** - Specifies whether the customer has dependents or not
- **Tenure** -  Duration of subscription in months
- **Phone Service** - if the customer has a phone service or not
- **MultipleLines** - If the customer has multiple lines or not
- **InternetService** - Customer's internet service provider 
- **OnlineSecurity** - If the customer has online security or not
- **OnlineBackup** - If the customer has online backup or not.''') 

    with col3:
        st.subheader('')
        st.subheader('')
        st.markdown('''
- **DeviceProtection** - If the customer has device protection or not
- **TechSupport** - If the customer has tech support or not
- **StreamingTV** - Whether the customer has streaming TV or not
- **StreamingMovies** - Whether the customer has streaming movies or not
- **Contract** - The contract term of the customer
- **PaperlessBilling** - Whether the customer has paperless billing or not
- **Payment Method** - The customer's payment method
- **MonthlyCharges** - Monthly charges to the customer 
- **TotalCharges** - Total amount charged to the customer
- **Churn** - Whether the customer churned or not. ''')



def get_data():
    # Function to get data from varying sources and display data IN DISPLAY DATA SECTION ON APP PAGE
    with st.form('Data entry Display'):
        st.subheader('Get Data')  # Row title
        selection = st.selectbox(label='Select churn data source', options=['In-built data', 'External data'], key='source_selection')  # data source options
        choice = st.form_submit_button('Select')

    if choice and selection == 'In-built data':
        # display internal data
        # write sql queries
        query1 = "SELECT * FROM dbo.LP2_Telco_churn_first_3000"
        # load data
        customer_df = pd.read_sql_query(query1, nection)
        with st.spinner("Dataset loading..."):
            time.sleep(3)
        st.subheader('View Data')
        st.write(customer_df.head(20))

    else:
        if selection == 'External data':
            file_upload = st.file_uploader(label = 'Choose a CSV file', type= 'csv', key = 'external data')
            if file_upload is not None:
                customer_df2 = pd.read_csv(file_upload)
                with st.spinner("Dataset loading..."):
                    time.sleep(3)
                st.subheader('View Data')
                st.dataframe(customer_df2.head(20))
        

if __name__ == "__main__":
    st.title('Customer Churn Data')
    get_data()
    detail()