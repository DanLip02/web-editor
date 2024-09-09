import pandas as pd
import streamlit as st
import re
from tools import SqlCreatorAlpha, Executor
from sqlalchemy import text
import dotenv
import os

dotenv.load_dotenv()
schema = os.getenv('schema')
pattern = r"(?<=\) )[\w+\s+]+(?=CONTEXT:)"
#pattern = r"(?<=\) )[\w+\s+]"

"""def connection_1(user: str = None, password: str = None, **kwargs):
    if user is None:
        user = os.getenv('user')
        # or raise KeyError('no user passed')
    if password is None:
        password = os.getenv('password')
    host = os.getenv('host')
    database = os.getenv('database')
    conn = psycopg2.connect(
        host=host,
        database=database,
        options="-c search_path=_defaults",
        user=user,
        password=password)
    return conn"""

def get_full_df(table, executor=None):
    if executor is None:
        executor: Executor = st.session_state['executor']
    sql = SqlCreatorAlpha(table=table, schema=schema)
    # sql.select().order_by('defaults_system_id', how='DESC')
    sql.select()
    # data = executor.execute(f"select * from {table} order by defaults_system_id desc")
    data = executor.execute(sql)
    return data.as_dataframe

def get_always_gen(table, executor=None):
    if executor is None:
        executor: Executor = st.session_state['executor']
    sql = SqlCreatorAlpha().from_raw(f"""select col.column_name from information_schema.columns col
    where col.identity_generation = 'ALWAYS' and col.table_schema not in('information_schema', 'pg_catalog')
        and col.table_name = '{table}'
            order by col.column_name""")
    data = executor.execute(sql)
    if 'Some error' in data:
        return None
    else:
        return data.as_dataframe.values.tolist()[0]

def get_default(table, executor=None):
    data_col = []
    data_val = []
    data_type = []
    if executor is None:
        executor: Executor = st.session_state['executor']
    # sql = SqlCreatorAlpha().from_raw(f"""select
    #     col.column_name,
    #     col.column_default,data_type
    #     from information_schema.columns col
    #     where col.column_default is not null
	#     and col.is_nullable = 'YES'
    #     and col.table_schema not in('information_schema', 'pg_catalog')
    #     and col.table_name = '{table}'
    #     order by col.column_name""")
    sql = SqlCreatorAlpha().from_raw(f"""select
            col.column_name,
            col.column_default,data_type
            from information_schema.columns col
            where col.column_default is not null
    	    and col.is_nullable = 'YES'
            and col.table_schema not in('information_schema', 'pg_catalog')
            and col.table_name = '{table}'
            order by col.column_name""")
    data = executor.execute(sql)
    data = data.as_dataframe.values.tolist()
    if data[0][0] == 'Some error':
        return None
    else:
        for val in data:
            data_col.append(val[0])
            if val[1] == 'true':
                val[1] = True
            elif val[1] == 'false':
                val[1] = False
            else:
                val[1] = val[1].split('::')[0].replace("'", "")
            data_val.append(val[1])
            data_type.append(val[2])
        return (data_col, data_val, data_type)

def get_nullable(table, executor=None):
    data_col = []
    data_val = []
    data_type = []
    if executor is None:
        executor: Executor = st.session_state['executor']
    sql = SqlCreatorAlpha().from_raw(f"""select
       col.column_name,
       col.column_default,data_type
       from information_schema.columns col
       where col.column_default is not null
	   and col.is_nullable = 'NO'
       and col.table_schema not in('information_schema', 'pg_catalog')
       and col.table_name = '{table}'
       order by col.column_name""")
    data = executor.execute(sql)
    data = data.as_dataframe.values.tolist()
    if data[0][0] == 'Some error':
        return None
    else:
        for val in data:
            data_col.append(val[0])
            if val[1] == 'true':
                val[1] = True
            elif val[1] == 'false':
                val[1] = False
            else:
                if val[1].split('::')[0].replace("'", "") != '':
                    val[1] = val[1].split('::')[0].replace("'", "")
                else:
                    val[1] = 'null'
            data_val.append(val[1])
            data_type.append(val[2])
        return (data_col, data_val, data_type)

def get_no_default(table, executor=None):
    data_col = []
    data_val = []
    data_type = []
    if executor is None:
        executor: Executor = st.session_state['executor']
    sql = SqlCreatorAlpha().from_raw(f"""select
       col.column_name,
       col.column_default,data_type
       from information_schema.columns col
       where col.column_default is null
	   and col.is_nullable = 'YES'
       and col.table_schema not in('information_schema', 'pg_catalog')
       and col.table_name = '{table}'
       order by col.column_name""")
    data = executor.execute(sql)
    data = data.as_dataframe.values.tolist()
    for val in data:
        data_col.append(val[0])
        data_type.append(val[2])
    return (data_col, data_type)

def get_df(table, executor=None):
    if executor is None:
        executor: Executor = st.session_state['executor']
    sql = SqlCreatorAlpha(table=table, schema=schema)
    # sql.select().order_by('defaults_system_id', how='DESC').limit(100)
    sql.select()
    data = executor.execute(sql)
    return data.as_dataframe

def get_date_df(time, executor: Executor = None) -> pd.DataFrame():
    if executor is None:
        executor = st.session_state['executor']
    sql = SqlCreatorAlpha(
        table=f"get_defaults(api=>'1.0', issuer_type_in=>'all',_date=>'{time} 16:59:59')",
        schema=schema
    ).select()
    df = executor.execute(sql).as_dataframe
    return df


def get_last_change(table: str, executor=None) -> pd.DataFrame():       #TODO  add condition to check if table_audit exist
    if executor is None:
        executor: Executor = st.session_state['executor']
    # query = f"SELECT * FROM _defaults._global_audit ORDER BY STAMP DESC LIMIT 5"
    query_test = f"""SELECT EXISTS (
    SELECT FROM 
        pg_tables
    WHERE 
        schemaname = '{schema}' AND 
        tablename  = '{table}_audit'
    );"""
    if executor.execute(query_test)[0][0] == True:
        sql = SqlCreatorAlpha(table=f'{table}_audit', schema=schema)
        sql.select().order_by('STAMP', how='DESC').limit(5)
        # df = pd.read_sql(query, conn)
        df = executor.execute(sql).as_dataframe
        return df

def insert_row(table, generator, executor=None):  # it's insert function based on create query and generator-dictionary from main
    columns = generator.keys()  # keys
    values = generator.values()  # value
    if executor is None:
        executor = st.session_state['executor']

    insert_row_query = SqlCreatorAlpha(table=table, schema=schema)
    insert_row_query.insert(generator)
    con = executor.engine.connect()

    try:
        con.begin()
        con.execute(text(repr(insert_row_query)))
        con.commit()
        # executor.execute(insert_row_query)
        # executor.engine.connect().execute(insert_row_query)
        return True

    except Exception as e:
        st.error(re.findall(pattern, e.args[0])[0])
        con.rollback()
        return False
    finally:
        con.close()

def update_table(table, inp_dict, df, ind, disable_,  executor=None):  # it's update query for db, based on iteration over dictionary
    if executor is None:
        executor: Executor = st.session_state['executor']

    sql = SqlCreatorAlpha(table=table, schema=schema)
    sql.update(inp_dict).where(f"{disable_} = {df[f'{disable_}'][ind]}")
    con = executor.engine.connect()

    try:
        con.begin()
        con.execute(text(repr(sql)))
        con.commit()
        return True

    except Exception as e:
        st.write(e)
        st.error(re.findall(pattern, e.args[0])[0])
        con.rollback()
        return False
    finally:
        con.close()

def delete_row(table, df, index, disable_, executor: Executor = None):
    if executor is None:
        executor = st.session_state['executor']
    sql = SqlCreatorAlpha(table=table, schema=schema)
    sql.delete().where(f"{disable_} = {df[f'{disable_}'][index]}")
    executor.execute(sql)
