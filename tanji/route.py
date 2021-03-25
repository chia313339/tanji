from flask import render_template, redirect, url_for, request, send_from_directory, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import os
import io
from bs4 import BeautifulSoup
import json
from datetime import datetime, date, timedelta
import numpy as np
import pandas
import matplotlib.pyplot as plt
import requests




def index():
    if "user" in session:
        user = session["user"]
        return render_template("index.html",user=user)
    else:
        return redirect(url_for("login"))

def login():
    if request.method == "POST":
        session.permanent = True
        user = request.form["id"]
        print(user)
        session["user"] = user
        return redirect(url_for("index"))
    else:
        if "user" in session:
            return redirect(url_for("index")) 
        return render_template("login.html")

def logout():
    session.pop("user", None)
    return redirect(url_for("login"))



# def get_stock_info(stock_no='2330'):
#     url = "https://goodinfo.tw/StockInfo/StockDetail.asp?STOCK_ID=%s" % (stock_no)
#     res = requests.get(url, headers = headers)
#     res.encoding = 'utf-8'
#     soup = BeautifulSoup(res.text, 'lxml')
#     data = soup.select_one('#txtFinBody')

def stock_info(stock_no):
    url = "https://goodinfo.tw/StockInfo/StockDetail.asp?STOCK_ID=%s" % (stock_no)
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36'  }
    res = requests.get(url, headers = headers)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'lxml')
    tmp = soup.select_one('.solid_1_padding_3_1_tbl').select('td')
    msg_list = []
    for msg in tmp:
        if msg.text.find('標的')>0:
            continue
        msg_list.append(msg.text)
    stock_info = {
        "stock_name":msg_list[0],
        "updated_date":msg_list[1],
        "st_price":msg_list[10],
        "open":msg_list[14],
        "high":msg_list[15],
        "low":msg_list[16],
        "vol":msg_list[26],
        "yd_price":msg_list[11],
        "diff_price":msg_list[12],
        "diff_pc":msg_list[13]
    }
    tmp = soup.select('.solid_1_padding_4_4_tbl')[2].select('td')
    stock_info["st_allname"]=tmp[1].text
    stock_info["st_cat"]=tmp[3].text
    stock_info["listed"]=tmp[5].text
    stock_info["f_value"]= tmp[7].text
    stock_info["capital"]=tmp[9].text
    stock_info["m_value"]=tmp[11].text
    stock_info["e_date"]=tmp[13].text
    stock_info["l_date"]=tmp[15].text
    stock_info["president"]=tmp[17].text
    stock_info["gm"]=tmp[19].text
    stock_info["speaker"]=tmp[21].text
    stock_info["ag"]=tmp[23].text
    stock_info["url"]=tmp[25].text
    stock_info["co_bond"]=tmp[29].text
    stock_info["private"]=tmp[31].text
    stock_info["special"]=tmp[33].text
    stock_info["business"]=tmp[35].text
    if float(stock_info['st_price'])>float(stock_info['yd_price']): 
        stock_info["trend"]='red'
    if float(stock_info['st_price'])<float(stock_info['yd_price']):
        stock_info["trend"]='green'
    if float(stock_info['st_price'])==float(stock_info['yd_price']): 
        stock_info["trend"]='black'
    
    return stock_info


def stock():
    if request.method == "POST":
        stock_no = request.form["stock_no"]
        print(stock_no)
        ans = stock_info(stock_no)

    else:
        stock_no = '2330' 
        print(stock_no)
        ans = stock_info(stock_no)
    return render_template('stock.html',stock_no=stock_no,ans=ans)


def tables():
    return render_template('tables.html')

def user():
    return render_template('user.html')

def register():
    return render_template('register.html')

def password():
    return render_template('password.html')

def p401():
    return render_template('p401.html')

def p404():
    return render_template('p404.html')

def p500():
    return render_template('p500.html')

def charts():
    return render_template('charts.html')
