from flask import render_template, redirect, url_for, request, send_from_directory, flash, jsonify
from flask import send_from_directory
import pandas as pd
import json
import psycopg2
import sqlalchemy
import datetime
from app.models.five_line import *
from app.models.stock_list import *

# 正式環境DB
pgdb_config={
'host':'34.81.198.238',
'port':5432,
'user':'tanji',
'password':'tanji',
'database':'tanji',
}

# 函數類
# 讀pgdb資料 
def get_data_from_pgdb(pgdb_config,sqls):
    conn = psycopg2.connect(**pgdb_config)
    cursor = conn.cursor()
    cursor.execute(sqls)
    result = cursor.fetchall()
    conn.commit()
    conn.close()
    return result




# 頁面類
def index():
    return render_template('index.html')

def stock_list():
    stock_list_stats_sql = '''SELECT * FROM stock_list_stats;'''
    stock_list_stats = get_data_from_pgdb(pgdb_config,stock_list_stats_sql)
    return render_template('stock_list.html',tstock_stat=stock_list_stats)

def five_line(stock_no='0050.TW'):
    print(datetime.datetime.now())
    stock_info = {
        'stock_no':stock_no,
        'end_dt':datetime.datetime.now().strftime("%Y-%m-%d"),
        'start_dt':(datetime.datetime.now() - datetime.timedelta(days=365*3.5)).strftime("%Y-%m-%d")
    }
    return five_line_view(stock_info['stock_no'], stock_info['start_dt'], stock_info['end_dt'])


def ml_stock():
    return render_template('ml_stock.html')

def page_not_found(e):
    return render_template('404.html'), 404