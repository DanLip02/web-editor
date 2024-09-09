import dotenv
import streamlit as st
import pandas as pd
from data_base import insert_row, update_table, delete_row, get_date_df, get_last_change, get_df, get_full_df, get_always_gen, get_default, get_nullable, get_no_default
from datetime import datetime as datetime_s
import datetime
import base64
from io import BytesIO
import os
from tools import Executor
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from typing import Any, Dict
import json
import numpy as np
import requests
global args

# todo           HELPFUL_LINKS
# https://github.com/streamlit/example-app-editable-dataframe/blob/main/streamlit_app.py with editable df
# https://discuss.streamlit.io/t/can-i-display-dash-datatables-on-streamlit/32459
# https://github.com/nadbm/react-datasheet
# https://medium.com/ssense-tech/streamlit-tips-tricks-and-hacks-for-data-scientists-d928414e0c16


#todo move to API as Misha did

# def connec(user: str = None, password: str = None, **kwargs) -> bool:
#     dotenv.load_dotenv()
#     if user is None:
#         # ser = os.getenv('user')
#         raise KeyError('no user passed')
#         # or raise KeyError('no user passed')
#     if password is None:
#         # password = os.getenv('password')
#         raise KeyError('no user passed')
#
#     host = os.getenv('host')
#     database = os.getenv('database')
#     port = os.getenv('port', '5432')
#     df_json = {'lg': f'{user}', 'pd': f'{password}'}
#     json_object = json.dumps(df_json)
#     r = requests.post('http://127.0.0.1:8000/api/connect_web/', data=json_object)
#     st.write(r.status_code)
#     if r.status_code == 200:
#         st.write(r, r.text)
#         try:
#             executor = Executor(
#                 host=host,
#                 database=database,
#                 port=port,
#                 user=user,
#                 password=password
#             )
#             st.session_state['executor'] = executor
#             return True
#         except ConnectionError:
#             return False
#     else:
#         return False

def connec(user: str = None, password: str = None, **kwargs) -> bool:
    dotenv.load_dotenv()
    if user is None:
        # ser = os.getenv('user')
        raise KeyError('no user passed')
        # or raise KeyError('no user passed')
    if password is None:
        # password = os.getenv('password')
        raise KeyError('no user passed')

    df_json = {'lg': f'{user}', 'pd': f'{password}'}
    json_object = json.dumps(df_json)
    port_api = os.getenv('port_api')    #port api in .env file
    host_api = os.getenv('host_api')    #host api in .env file
    if "api" not in st.session_state:
        st.session_state.api = {'host_api': host_api, 'port_api': port_api}
    r = requests.post(f'http://{host_api}:{port_api}/api/connect_web/', data=json_object)
    if r.status_code == 200:
        if json.loads(r.text) == 'Error':
            return False
        else:
            executor = Executor(
                                host=json.loads(r.text)['host'],
                                database=json.loads(r.text)['db'],
                                port=json.loads(r.text)['port'],
                                user=user,
                                password=password
                            )
            st.session_state['executor'] = executor
            return True
def login(userName: str, password: str):
    if (userName is None):
        return False
    args = [userName, password]
    if 'executor' in st.session_state:
        return True
    else:
        result = connec(userName, password)
        if not result:
            st.error('Incorrect Username/Password. Try again')
        else:
            return result

def show_main_page():
    with mainSection:
        st.write(f'Welcome, {list(st.session_state.user.keys())[0]}!')
        global args
        args = (list(st.session_state.user.keys())[0], list(st.session_state.user.values())[0])
        main_operation()

def logout_click():
    del st.session_state.storage_add
    del st.session_state.storage_upd
    del st.session_state.storage_del
    del st.session_state.temp_add
    del st.session_state.temp_upd
    del st.session_state['executor']
    st.session_state['loggedIn'] = False

def show_logout_page():
    loginSection.empty()
    with logOutSection:
        st.sidebar.button("Log Out", key="logout", on_click=logout_click)

def login_click(userName, password):
    if login(userName, password):
        st.session_state['loggedIn'] = True
        st.session_state.user = {f'{userName}': f'{password}'}
    else:
        st.session_state['loggedIn'] = False

def show_login_page():
    with loginSection:
        if st.session_state['loggedIn'] == False:
            global userName
            global password
            userName = st.text_input(label="", placeholder="Enter your user name")
            password = st.text_input(label="", placeholder="Enter password", type="password")
            st.button("Login", on_click=login_click, args=(userName, password))

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def dataframe_explorer(df: pd.DataFrame, case: bool = True) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
        case (bool, optional): If True, text inputs will be case sensitive. Defaults to True.
    Returns:
        pd.DataFrame: Filtered dataframe
    """

    random_key_base = pd.util.hash_pandas_object(df)

    df = df.copy()

    # Try to convert datetimes into standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d')
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect(
            "Filter dataframe on",
            df.columns,
            key=f"{random_key_base}_multiselect",
        )
        filters: Dict[str, Any] = dict()
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 100:          #Todo think about all unique objects
                left.write("↳")
                filters[column] = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                    key=f"{random_key_base}_{column}",
                )
                df = df[df[column].isin(filters[column])]
            elif is_numeric_dtype(df[column]):
                left.write("↳")
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                filters[column] = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                    key=f"{random_key_base}_{column}",
                )
                df = df[df[column].between(*filters[column])]
            elif is_datetime64_any_dtype(df[column]):
                left.write("↳")
                filters[column] = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                    key=f"{random_key_base}_{column}",
                )
                if len(filters[column]) == 2:
                    filters[column] = tuple(map(pd.to_datetime, filters[column]))
                    start_date, end_date = filters[column]
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                left.write("↳")
                filters[column] = right.text_input(
                    f"Pattern in {column}",
                    key=f"{random_key_base}_{column}",
                )
                if filters[column]:
                    df = df[df[column].str.contains(filters[column], case=case)]
    return df

def insert(data: list, table ,nullable_col: list, nullable_val: list):
    result = []
    from_json = {}
    if len(data) > 0:
        # st.write(list(data[0].keys()))
        for row in data:
            for col in list(row.keys()):
                # st.write(col)
                if nullable_col is not None:
                    for i, col_def in enumerate(nullable_col):
                        if col == col_def and (row[col] == "" or row[col] is None):
                            row[col] = nullable_val[i]
        for row in data:
             try:
                 del row['_index']
             except:
                 pass
             # if insert_row(table, row) == False:
             #    result.append(row)
        for row in data:
            temp_ = {}
            for col in row.keys():
                if col != 'ogrn' and col != 'publication_year' and col != 'report_date':
                    temp_[col] = row[col]

                if isinstance(row[col], datetime.date):
                    row[col] = row[col].strftime('%Y-%m-%d')
            from_json['values'] = [({'publication_year' : row['publication_year'], 'ogrn' : row['ogrn'], 'report_date' : row['report_date'], 'values' : temp_})]
            userName = list(st.session_state.user.keys())[0]
            password = st.session_state.user[userName]
            st.write(from_json) #TODO get jaon for tests api
            from_json['lg'] = f'{userName}'
            from_json['pd'] = f'{password}'
            json_object = json.dumps(from_json)
            # Print JSON object
            #todo change to api function (@app.post('/api/insert_fp/')) using requests and port
            # results = put_matrix(json_object)
            host_api = st.session_state.api['host_api']
            port_api = st.session_state.api['port_api']
            r = requests.post(f'http://{host_api}:{port_api}/api/insert_fp_web/', data=json_object)
            if r.status_code == 200:
                results = json.loads(r.text)

                if len(results['duplicate_rows']) != 0:
                    st.error('A record with such data exists')
                    for row_data in results['duplicate_rows']:
                        result.append(row_data)

                if len(results['error_rows']) != 0:
                    st.error('Report date can not be more than publication_year. Or another error. Please, check!')
                    for row_data in results['error_rows']:
                        result.append(row_data)

    return result

def update_bd(data: list, table: str, df: pd.DataFrame, index_mass: list, disable_: str, nullable_col: list, nullable_val: list):
    result = []
    from_json = {}
    if len(data) > 0:
        for row in data:
            for col in list(row.keys()):
                # st.write(col)
                if nullable_col is not None:
                    for i, col_def in enumerate(nullable_col):
                        if col == col_def and (row[col] == "" or row[col] is None):
                            row[col] = nullable_val[i]
        for index, cell in enumerate(data):
            temp_ = {}
            ind = index_mass[index]
            for col in cell.keys():
                if col == 'year' or col == 'publication_year':
                    cell[col] = int(cell[col])
                else:
                    if type(cell[col]) == float or type(cell[col]) == int:
                        cell[col] = str(int(cell[col]))
                    if isinstance(cell[col], np.integer):
                        cell[col] = str(int(cell[col]))
                    if isinstance(cell[col], np.floating):
                        cell[col] = str(float(cell[col]))
                    if isinstance(cell[col], datetime.date):
                        cell[col] = cell[col].strftime('%Y-%m-%d')
            # if update_table(table, cell, df, ind, disable_) == False:
            #     result.append(cell)
            # cell['year'] = int(cell['year'])
            # cell['publication_year'] = int(cell['publication_year'])
            from_json['values'] = [{'id': str(df[disable_][ind]), 'values': cell}]
            userName = list(st.session_state.user.keys())[0]
            password = st.session_state.user[userName]
            from_json['lg'] = f'{userName}'
            from_json['pd'] = f'{password}'
            json_object = json.dumps(from_json)
            # Print JSON object
            # results = update_matrix(json_object)
            # st.write(results)
            host_api = st.session_state.api['host_api']
            port_api = st.session_state.api['port_api']
            r = requests.post(f'http://{host_api}:{port_api}/api/update_fp_web/', data=json_object)
            st.write(r.status_code)
            if r.status_code == 200:
                results = json.loads(r.text)

                if len(results['duplicate_rows']) != 0:
                    st.error('A record with such data exists')
                    for row_data in results['duplicate_rows']:
                        # st.write(row_data['values'], type(row_data), row_data, row_data.keys())
                        result.append(row_data['values'])

                if len(results['error_rows']) != 0:
                    st.error('Report date can not be more than publication_year. Or another error. Please, check!')
                    for row_data in results['error_rows']:
                        # st.write(row_data['values'], type(row_data), row_data, row_data.keys())
                        result.append(row_data['values'])

    #st.write(result)
    return result

def delete_data(data: list, table, df, disable_: str):
    if len(data) > 0:
        from_json = {}
        for index in data:
            if 'id' not in from_json:
                from_json['id'] = []
            from_json['id'].append(index)

        userName = list(st.session_state.user.keys())[0]
        password = st.session_state.user[userName]
        from_json['lg'] = f'{userName}'
        from_json['pd'] = f'{password}'
        json_object = json.dumps(from_json)
        host_api = st.session_state.api['host_api']
        port_api = st.session_state.api['port_api']
        r = requests.post(f'http://{host_api}:{port_api}/api/delete_fp_web/', data=json_object)
        if r.status_code == 200:
            st.write('Success')
            # delete_row(table, df, index, disable_)

def parser_update_2_0(df):
    result = []  # array of dict
    index_mass = []  # array of rows, where edited cells
    df = df.reset_index()
    if len(list(st.session_state['data_editor']['edited_rows'].keys())) > 0:
        for index in list(st.session_state['data_editor']['edited_rows'].keys()):
            buf_dict = {}  # dict for appending to array (result)
            index_mass.append(df[df.index == index]['index'].values[0])
            buf_df = df[df.index == index]
            del buf_df['index']
            for col in buf_df:
                # if col != 'index':
                    if col not in buf_dict:
                        if buf_df[col].dtype == 'datetime64[ns]':
                            buf_df = buf_df.astype({col : str})
                        if buf_df[col].values[0] == 'NaT':
                            buf_df[col].values[0] = 'null'
                        buf_dict[col] = buf_df[col].values[0]
            if len(buf_dict) > 0:
                result.append(buf_dict)
    if len(result) > 0:
        return (result, index_mass)
    else:
        return ([], [])

@st.cache_data
def store_add(storage_add, added_rows):
    if "storage_add" not in st.session_state:
        st.session_state.storage_add = storage_add
    st.session_state.storage_add += added_rows
    return st.session_state.storage_add

#Our flag(NCR)
def flag():
    file_ = open(r"NCR.gif", "rb")
    # file_ = open(r"NCR.gif", "rb")
    # file_ = open(r"/Users/lipatovaelena/PycharmProjects/defaults-table-web-editor/NCR.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="flag gif" style="float:right;width:200px;height:200px;margin-right:-300px;margin-top:-60px">',
        unsafe_allow_html=True,
    )

def redact_insert(data: list) -> pd.DataFrame():
    result = pd.DataFrame()
    for obj in data:
        result = pd.concat([result, pd.DataFrame.from_dict(obj)], axis=1)
    return result

# specific config of columns for editable df. Types of data got from db
#todo move to API
def config_gen(default:list, nullable:list, default_type:list, nullable_type:list, default_val:list , nullable_val:list, no_default_col:list, no_default_type:list) -> dict:
    config = {}
    if default is not None:
        for i, col in enumerate(default):
            if col not in config:
                config[col] = []
            if default_type[i] == 'date':
                config[col] = st.column_config.DateColumn(
                    col,
                    min_value=datetime.date(1900, 1, 1),
                    max_value=datetime.date(2050, 1, 1),
                    format="YYYY-MM-DD",
                    default=default_val[i],
                    step=1
                )
            elif default_type[i] == 'boolean':
                config[col] = st.column_config.CheckboxColumn(
                    col,
                    default=default_val[i],
                )
            elif default_type[i] == 'text':
                    config[col] = st.column_config.TextColumn(
                    col,
                    default=default_val[i]
                )
            elif default_type[i] == 'double precision':
                    config[col] = st.column_config.NumberColumn(
                        col,
                        required=True,
                        default=default_val[i]
                    )
            elif default_type[i] == 'bigint':
                    config[col] = st.column_config.NumberColumn(
                        col,
                        required=True,
                        default=default_val[i],
                        format='%d'
                    )
    if nullable is not None:
        for i, col in enumerate(nullable):
            if col not in config:
                config[col] = []
            if nullable_type[i] == 'date':
                config[col] = st.column_config.DateColumn(
                    col,
                    min_value=datetime.date(1900, 1, 1),
                    max_value=datetime.date(2050, 1, 1),
                    format="YYYY-MM-DD",
                    required= True,
                    default=datetime_s.strptime(nullable_val[i], '%Y-%m-%d').date(),
                    step=1
                )
            elif nullable_type[i] == 'boolean':
                config[col] = st.column_config.CheckboxColumn(
                    col,
                    required=True,
                    default=nullable_val[i],
                )
            elif nullable_type[i] == 'text':
                config[col] = st.column_config.TextColumn(
                    col,
                    required=True,
                    default= nullable_val[i]
                )
            elif nullable_type[i] == 'double precision':
                    config[col] = st.column_config.NumberColumn(
                        col,
                        required=True,
                        default=nullable_val[i]
                    )
            elif nullable_type[i] == 'bigint':
                    config[col] = st.column_config.NumberColumn(
                        col,
                        required=True,
                        default=nullable_val[i],
                        format='%d'
                    )

    if no_default_type is not None:
        for i, col in enumerate(no_default_col):
            if col not in config:
                config[col] = []
            if no_default_type[i] == 'date':
                config[col] = st.column_config.DateColumn(
                    col,
                    min_value=datetime.date(1900, 1, 1),
                    max_value=datetime.date(2050, 1, 1),
                    format="YYYY-MM-DD",
                    default=datetime.date(1900, 1, 1),
                    step=1
                )
            elif no_default_type[i] == 'boolean':
                config[col] = st.column_config.CheckboxColumn(
                    col,
                    default=False,
                )
            elif no_default_type[i] == 'text':
                    config[col] = st.column_config.TextColumn(
                        col,
                        default= 'null'
                        )

            elif no_default_type[i] == 'double precision':
                    config[col] = st.column_config.NumberColumn(
                        col,
                        default=None
                    )
            elif no_default_type[i] == 'json':
                    config[col] = st.column_config.Column(
                        col,
                    )
            elif no_default_type[i] == 'bigint':
                    config[col] = st.column_config.NumberColumn(
                        col,
                        default=None,
                        format='%d'
                    )

    return config

#support func for change types of columns. Necessary with dataframe_explorer
def check_null(data: pd.DataFrame(), default:list, nullable:list, default_type:list, nullable_type:list, no_default_col:list, no_default_type:list):
    for col in data.columns:
        if data[col].dtype == 'datetime64[ns]':
            if default is not None:
                for i, col_ in enumerate(default):
                    if col == col_ and default_type[i] == 'text':
                        data = data.astype({col : str})
            if nullable is not None:
                for i, col_ in enumerate(nullable):
                    if col == col_ and nullable_type[i] == 'text':
                        data = data.astype({col : str})
            if no_default_col is not None:
                for i, col_ in enumerate(no_default_col):
                    if col == col_ and no_default_type[i] == 'text':
                        data = data.astype({col : str})
    return data.replace('NaT', 'null')

# Support func for convert datetime to str for requests to db.
def convert_redact_df(data, default:list, nullable:list, default_type:list, nullable_type:list, no_default_col:list, no_default_type:list):
    for values in data:
        for col in values:
            if type(values[col]) == str:
                # st.write(col, values[col])
                if default is not None:
                    for i, col_ in enumerate(default):
                        if col == col_ and default_type[i] == 'date':
                            values[col] = datetime_s.strptime(values[col], '%Y-%m-%d').date()
                if nullable is not None:
                    for i, col_ in enumerate(nullable):
                        if col == col_ and nullable_type[i] == 'date':
                            values[col] = datetime_s.strptime(values[col], '%Y-%m-%d').date()
                if no_default_col is not None:
                    for i, col_ in enumerate(no_default_col):
                        if col == col_ and no_default_type[i] == 'date':
                            values[col] = datetime_s.strptime(values[col], '%Y-%m-%d').date()
    return data

def main_operation():
    dotenv.load_dotenv()
    table = os.getenv('table')  # table to work
    host_api = st.session_state.api['host_api']
    port_api = st.session_state.api['port_api']
    userName = list(st.session_state.user.keys())[0]
    password = st.session_state.user[userName]
    df_json = {'table': table, 'lg': f'{userName}', 'pd': f'{password}'}
    json_object = json.dumps(df_json)
    # total_df = pd.DataFrame()  # final df
    last_change_df = pd.DataFrame()  # df for all rows by date
    # df = get_df(table)  # df form postgres
    # compare_df = get_full_df(table)
    r = requests.post(f'http://{host_api}:{port_api}/api/get_fp_web/', data=json_object)
    df = pd.DataFrame() #base
    compare_df = pd.DataFrame()
    if r.status_code == 200:
        df = pd.DataFrame().from_dict(json.loads(r.text))
        # st.write(pd.DataFrame().from_dict(df))
        compare_df = pd.DataFrame().from_dict(json.loads(r.text))
    default_col, default_val, default_type = None, None, None
    nullable_col, nullable_val, nullable_type = None, None, None
    disable_ = None
    if get_always_gen(table) is not None:
        disable_ = get_always_gen(table)
    if get_default(table) is not None:
        default_col, default_val, default_type = get_default(table)
    if get_nullable(table) is not None:
        nullable_col, nullable_val, nullable_type = get_nullable(table)
    no_default_col, no_default_type = get_no_default(table)
    storage_add = []        #array to store added rows to not lost them
    storage_upd = []        #array to store updated rows to not lost them
    temp_add = []
    temp_upd = []
    storage_upd_ind = []
    storage_del = []        #array to store deleted rows to not lost them
    dubl = []
    check = []
    df_to_edit = pd.DataFrame()
    redact_df_add = pd.DataFrame()
    redact_df_upd = pd.DataFrame()
    # date_columns = []

    if "redact_df_add" not in st.session_state:
        st.session_state.redact_df_add = redact_df_add
    if "redact_df_upd" not in st.session_state:
        st.session_state.redact_df_upd = redact_df_upd
    if "compare_df" not in st.session_state:
        st.session_state.compare_df = compare_df
    if "dubl" not in st.session_state:
        st.session_state.dubl = dubl
    if "check" not in st.session_state:
        st.session_state.check = check
    if "storage_add" not in st.session_state:
        st.session_state.storage_add = storage_add
    if "storage_upd" not in st.session_state:
        st.session_state.storage_upd = storage_upd
    if "storage_upd_ind" not in st.session_state:
        st.session_state.storage_upd_ind = storage_upd_ind
    if "storage_del" not in st.session_state:
        st.session_state.storage_del = storage_del
    if "temp_add" not in st.session_state:
        st.session_state.temp_add = temp_add
    if "temp_upd" not in st.session_state:
        st.session_state.temp_upd = temp_upd

    date_df = st.sidebar.checkbox('Get full table by specific date')      #checkbox for getting df by date
    if date_df:
        time = st.sidebar.date_input('Enter the day you want to see', value=datetime.datetime.now())  # choosing date
        last_change_df = get_date_df(time)
        st.write(f'Table by the {time}')
        st.write(last_change_df)  # writting df

    but = st.sidebar.radio('Choose rows to display', ('Specific number of rows sorted by addition date','All rows'))
    num = 0  # number os rows

    if but == 'All rows':
        df = get_full_df(table)  # get num == all rows at df
        num = len(df)
    elif but == 'Specific number of rows sorted by addition date':
        value = 100
        if len(df) < 100:
            value = len(df)
        num = st.sidebar.number_input("Disply number of rows", 1, len(df), value=value)

    df_for_ogrn = df.iloc[:num]  # temp data_frame for n-rows
    if disable_ is not None:
        df_to_edit = df_for_ogrn.drop(columns=disable_).sort_index().reset_index(drop=True)
    else:
        df_to_edit = df_for_ogrn.sort_index().reset_index(drop=True)

    # df_to_edit = df_to_edit
    df_explorer = check_null(dataframe_explorer(df_to_edit, case=True),default_col, nullable_col, default_type, nullable_type, no_default_col, no_default_type)
    copy_ = df_explorer
    config_test = config_gen(default_col, nullable_col, default_type, nullable_type, default_val, nullable_val, no_default_col, no_default_type)
    edit_df = st.data_editor(df_explorer,
                            num_rows='dynamic',
                            key="data_editor",
                            column_config= config_test)

    st.sidebar.write('Length of table with all rows:', len(compare_df))
    st.sidebar.write('Length of table with filtred rows:', len(edit_df))
    st.sidebar.write('Length of table by specific date:', len(last_change_df))

    if get_last_change(table) is not None:
        st.sidebar.write('Last table change', get_last_change(table))

    show_add = st.sidebar.checkbox('Display added rows')
    show_upd = st.sidebar.checkbox('Display updated rows')
    show_del = st.sidebar.checkbox('Display deleted rows')

    # container = []
    # container_dubl = []
    # if len(st.session_state['data_editor']['added_rows']) > 0 or len(                   #In case to check not update or not insert
    #         st.session_state['data_editor']['edited_rows']) > 0:
    #     container = [checker(edit_df, df, parser_insert(edit_df), dubl, check), checker(edit_df, df, parser_update(edit_df)[0], dubl, check)]     #contains result of checker for update-insert."""
    #     container_dubl = [checker_full(st.session_state.compare_df, st.session_state.compare_df, (edit_df), dubl, check), checker_full(st.session_state.compare_df, st.session_state.compare_df, parser_update(edit_df)[0], dubl, check)]

    if show_add:
        st.write('Added rows:')
        st.write(st.session_state.storage_add)
    if show_upd:
        st.write('Updated rows:')
        st.write(st.session_state.storage_upd)
    if show_del:
        st.write('Deleted rows:')
        st.write(st.session_state.storage_del)
    if st.sidebar.button('Reset'):              #if deleted important rows
        del st.session_state.storage_add
        del st.session_state.storage_upd
        del st.session_state.storage_del
        del st.session_state.temp_add
        del st.session_state.temp_upd
        st.experimental_rerun()

    if st.button('Prepare to submit'):
        st.session_state.storage_add += st.session_state['data_editor']['added_rows']
        st.session_state.storage_upd += parser_update_2_0(edit_df)[0]
        st.session_state.storage_upd_ind += parser_update_2_0(edit_df)[1]
        for val in copy_[~copy_.index.isin(edit_df.index)].index.tolist():
            st.session_state.storage_del.append(val)

    if st.button('Submit'):
        st.session_state.temp_add = []
        st.session_state.temp_upd = []
        st.session_state.temp_upd = update_bd(st.session_state.storage_upd, table, df_for_ogrn, st.session_state.storage_upd_ind, disable_[0], nullable_col, nullable_val)
        st.session_state.temp_add = insert(st.session_state.storage_add, table, nullable_col , nullable_val)
        delete_data(st.session_state.storage_del, table, df_for_ogrn, disable_[0])
        st.session_state.storage_del = []

        if len(st.session_state.temp_add) == 0:
            #update_bd(st.session_state.storage_upd, table, df_for_ogrn, st.session_state.storage_upd_ind)     # todo error is here
            st.session_state.storage_add = []

        if len(st.session_state.temp_upd) == 0:
            st.session_state.storage_upd = []

        if len(st.session_state.temp_upd) == 0 and len(st.session_state.temp_add) == 0 and len(st.session_state.storage_del) == 0:
            st.write('OK')
            st.cache_data.clear()
            st.cache_resource.clear()
            st.experimental_rerun()

    st.warning(f'This is table with errors of added rows. '
               'If it is not "empty", please redact')

    if len(st.session_state.temp_add) > 0:
        redact_df_add = st.data_editor(convert_redact_df(st.session_state.temp_add, default_col, nullable_col, default_type, nullable_type,
                   no_default_col, no_default_type), key='add')
    # for val in redact_df_add:
    #     for v, k in val.items():
    #         st.write(type(k), k, v)

    if len(st.session_state.temp_add) != 0:
        st.session_state.storage_add = []

        st.session_state.storage_add = redact_df_add

    st.warning('This is table with errors of updated rows. '
               'If it is not "empty", please redact')

    if len(st.session_state.temp_upd) > 0:
        redact_df_upd = st.data_editor(st.session_state.temp_upd, key='upd')

    if len(st.session_state.temp_upd) != 0:
        st.session_state.storage_upd = []
        st.session_state.storage_upd = redact_df_upd

    date = datetime.datetime.now()

    if len(last_change_df) > 0:
        last_change_df = last_change_df
    else:
        last_change_df = df

    csv = last_change_df.to_csv().encode('utf-8')

    st.download_button(label='📥 Download table as xlsx',
                       data=to_excel(last_change_df),
                       file_name=f'default_table_({date}).xlsx')
    st.download_button(
        label="📥 Download data as CSV",
        data=csv,
        file_name=f'default_table_({date}).csv',
        mime='text/csv',
    )

    if st.sidebar.button('HELP'):
        with open('manual_dt/How to use server default table_.docx', 'rb') as f:
            st.download_button(
                label="📥 Download help document dock",
                data=f,
                file_name=f'How to use server default table_.docx',
            )

        with open('manual_dt/Example-server.xlsx', 'rb') as f:
            st.download_button(
                label="📥 Download example of xlsx file",
                data=f,
                file_name=f'Example-server.xlsx',
            )



if __name__ == '__main__':
    flag()
    headerSection = st.container()
    mainSection = st.container()
    loginSection = st.container()
    logOutSection = st.container()

    with headerSection:
        # first run will have nothing in session_state
        if 'loggedIn' not in st.session_state and 'user' not in st.session_state:
            st.session_state['loggedIn'] = False
            st.session_state.user = {}
            show_login_page()
        else:
            if st.session_state['loggedIn']:
                show_main_page() 
                show_logout_page()
            else:
                show_login_page()
