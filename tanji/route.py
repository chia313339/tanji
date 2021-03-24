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
    data = BeautifulSoup(res.text, 'lxml').select_one('.solid_1_padding_3_1_tbl').select('td')
    stock_info = {
        "stock_name":data[1].text.replace('\xa0', ' ')
    }
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
