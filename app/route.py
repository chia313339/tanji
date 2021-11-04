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
    sqls = '''SELECT stock_no, stock_name, price, ROUND( dividend::numeric, 2 ) as dividend, yield
	FROM public.stock_list_stats
	where per < 20
	and pbr <2
	and l10y_div_cnt >5
	and (five_line_2y like '%標準差之外相對低點%' or five_line_1y like '%標準差之外相對低點%')
	and stock_no in ('2330','2454',	'2317',	'2303',	'2881',	'2308',	'1301',	'1303',	'2882',	'2412',	'2891',	'2002',	'2886',	'5871',	'2884',	'3711',	'1216',	'6415',	'2885',	'2603',	'1326',	'1101',	'2357',	'2892',	'2379',	'3034',	'5880',	'2382',	'3008',	'2880',	'2615',	'2887',	'2207',	'2327',	'2409',	'2609',	'2912',	'3045',	'2395',	'5876',	'4938',	'6505',	'1590',	'1402',	'2801',	'8046',	'8454',	'4904',	'9910',	'2408')
	order by yield desc'''
    recommed_tstock = get_data_from_pgdb(pgdb_config,sqls)
    return render_template('index.html', recommed_tstock=recommed_tstock)

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