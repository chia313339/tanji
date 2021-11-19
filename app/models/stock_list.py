from flask import request, jsonify, render_template, redirect, url_for, make_response
import pandas as pd
import numpy as np
import requests
from scipy.stats import norm
from scipy import stats
from sklearn.linear_model import LinearRegression
from datetime import datetime
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from yahoo_fin.stock_info import *
import pyecharts.options as opts
from pyecharts.faker import Faker
from pyecharts.charts import Kline, Line, Bar, Grid
from jinja2 import Markup
import psycopg2
import sqlalchemy
from sqlalchemy import create_engine

# 正式環境DB
pgdb_config={
'host':'35.194.216.83',
'port':5432,
'user':'tanji',
'password':'tanji',
'database':'tanji',
}

# 執行pgdb的SQL，轉換dataframe
def sql_to_df(pgdb_config,sqls):
    engine = create_engine("postgresql://%s:%s@%s:5432/%s" % (pgdb_config['user'],pgdb_config['password'],pgdb_config['host'],pgdb_config['database']))
    df = pd.read_sql_query(sqls,con=engine)
    return df


def get_stock_list_csv():
    sqls = '''SELECT stock_no, stock_name, price, ROUND( dividend::numeric, 2 ) as dividend, dividend_st, yield_ch, yield, dividend_p, l3y_div_avg, l6y_div_avg, l10y_div_avg, l10y_div_cnt, q1eps, q2eps, q3eps, q4eps, acc_eps, l1y_eps, eps_goal, eps_eva, per, pbr, five_line_1y, five_line_2y, five_line_3y, ex_div_dt, record_dt, pay_dt, hd, sh_num, sh_pay, update_time FROM public.stock_list_stats'''
    df = sql_to_df(pgdb_config,sqls)
    df.columns = ['代號', '公司', '股價', '配息', '配股', '現金殖利率', '殖利率',
       '配息率', '3年平均股利', '6年平均股利', '10年平均股利', '10年股利次數', '1QEPS',
       '2QEPS', '3QEPS', '4QEPS', '今年累積EPS', '去年EPS', '去年EPS達成率', '獲利表現佳', '本益比', '股價淨值比', 'five_line_1y', 'five_line_2y', 'five_line_3y', '除息日', '除權日', '發息日', '董監持股', '多少張以上要繳健保費',
       '一張繳健保費','update_time']
    resp = make_response(df.to_csv(encoding='utf-8', index=False))
    resp.headers["Content-Disposition"] = "attachment; filename=stock.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp

def get_ustock_list_csv():
    sqls = '''SELECT stock_no, stock_name, price, dividend, yield_, stock_status_l1y, stock_l1y_r2, stock_l1y_slope, stock_l1y_sd, stock_status_l2y, stock_status_l3y, update_time FROM public.ustock_table;'''
    df = sql_to_df(pgdb_config,sqls)
    df.columns = ['代號',	'名稱',	'價格',	'配息',	'現金殖利率',	'近1年價位', '回歸解釋率', '斜率', '標準差',	'近2年價位',	'近3年價位', 'update_time']
    resp = make_response(df.to_csv(encoding='utf-8', index=False))
    resp.headers["Content-Disposition"] = "attachment; filename=ustock.csv"
    resp.headers["Content-Type"] = "text/csv"
    return resp