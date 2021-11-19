from flask import render_template, redirect, url_for, request, send_from_directory, flash, jsonify
from flask import send_from_directory
import pandas as pd
import json
import psycopg2
import sqlalchemy
import datetime
from app.models.base import *
from app.models.five_line import *
from app.models.stock_list import *
from app.models.line import *




# 頁面類
def index():
    sqls = '''SELECT stock_no, stock_name, price, ROUND( dividend::numeric, 2 ) as dividend, yield
	FROM public.stock_list_stats
	where per < 20
	and pbr <2
	and l10y_div_cnt >7
    and split_part(yield,'%',1)::numeric >3
    and eps_eva = 'true'
    and (five_line_1y like '%低點%' or five_line_2y like '%低點%')
	order by yield desc,eps_goal desc'''
    recommed_tstock = get_data_from_pgdb(pgdb_config,sqls)
    sp = "('"+"','".join(tickers_sp500())+"')"
    sqls2 = '''SELECT stock_no, stock_name, price, dividend, yield_, stock_status_l1y, stock_l1y_r2, stock_l1y_slope, stock_l1y_sd, stock_status_l2y, stock_status_l3y, update_time FROM public.ustock_table where stock_l1y_slope > 0.5 and stock_l1y_r2 > 0.7 and (stock_status_l2y like '%絕對低點%' or stock_status_l2y like '%標準差之外相對低點%' or stock_status_l1y like '%絕對低點%' or  stock_status_l1y like '%標準差之外相對低點%') and stock_no in '''
    sqls2 = sqls2 + sp + "order by stock_l1y_slope desc"
    recommed_ustock = get_data_from_pgdb(pgdb_config,sqls2)
    return render_template('index.html', recommed_tstock=recommed_tstock,recommed_ustock=recommed_ustock)

def stock_list():
    stock_list_stats_sql = '''SELECT * FROM stock_list_stats;'''
    stock_list_stats = get_data_from_pgdb(pgdb_config,stock_list_stats_sql)
    return render_template('stock_list.html',tstock_stat=stock_list_stats)

def us_stock_list():
    us_stock_list_stats_sql = '''SELECT * FROM ustock_table;'''
    us_stock_list_stats = get_data_from_pgdb(pgdb_config,us_stock_list_stats_sql)
    return render_template('us_stock_list.html',ustock_stat=us_stock_list_stats)

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