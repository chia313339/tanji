from flask import request, jsonify, render_template, redirect, url_for
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


# 爬取goodinfo台股資訊
def get_tstock_info(tstock_id):
    url = "https://goodinfo.tw/StockInfo/StockDetail.asp?STOCK_ID=%s" % tstock_id
    user_agent = UserAgent()
    headers = {'user-agent': user_agent.random}
    # 獲取 html 資訊
    res = requests.get(url, headers = headers)
    res.encoding='utf-8'
    if res.status_code == 200:
        tmp = BeautifulSoup(res.text, 'lxml')
        # 股票名稱
        stock_name = tmp.find_all('table', class_ = 'b1 p4_2 r10')[0].select('tr a')[0].text.replace(u'\xa0', u' ')
        # 公司名稱
        try:
            com_name = tmp.find_all('table', class_ = 'b1 p4_4 r10')[0].select('tr td')[2].text
        except:
            com_name = ""
        # 產業別
        try:
            com_class = tmp.find_all('table', class_ = 'b1 p4_4 r10')[0].select('tr td')[4].text
        except:
            com_class = ""
        # 上市櫃
        try:
            listed = tmp.find_all('table', class_ = 'b1 p4_4 r10')[0].select('tr td')[6].text
        except:
            listed = ""
        # 資本額
        try:
            capital = tmp.find_all('table', class_ = 'b1 p4_4 r10')[0].select('tr td')[10].text
        except:
            capital = ""
        # 市值
        try:
            mvalue = tmp.find_all('table', class_ = 'b1 p4_4 r10')[0].select('tr td')[12].text
        except:
            mvalue = ""
        # 股利分配狀況
        try:
            div_stat = tmp.find_all('table', class_ = 'b1 p4_0 r10 row_bg_2n row_mouse_over')[0].select('tr td')[-1].text.replace(u'\xa0', u' ').replace(u'\u2002', u' ').split(' ')[1]
        except:
            div_stat = ""
        # 最後一次發放年度
        try:
            lsy_div = tmp.find_all('table', class_ = 'b1 p4_0 r10 row_bg_2n row_mouse_over')[0].select('tr td')[5].text
        except:
            lsy_div = ""
        # 現金股利
        try:
            dividend = tmp.find_all('table', class_ = 'b1 p4_0 r10 row_bg_2n row_mouse_over')[0].select('tr td')[6].text
        except:
            dividend = ""
        # 現金殖利率
        try:
            div_yield = round(float(dividend)/float(tmp.find_all('table', class_ = 'b1 p4_2 r10')[0].select('tr td')[5].text),4)
        except:
            div_yield = ""
        # 更新時間
        try:
            update_time = tmp.find_all('table', class_ = 'b1 p4_2 r10')[0].select('tr td')[4].text.split(' ')[1]
        except:
            update_time = "非台股無詳細資料"
        
    else:
        stock_name = ""
        com_name = ""
        com_class = ""
        listed = ""
        capital = ""
        mvalue = ""
        div_stat = ""
        lsy_div = ""
        dividend = ""
        div_yield = ""
        update_time = "非台股無詳細資料"

    tstock_info = {
        'stock_name':stock_name,
        'com_name':com_name,
        'com_class':com_class,
        'listed':listed,
        'capital':capital,
        'mvalue':mvalue,
        'div_stat':div_stat,
        'lsy_div':lsy_div,
        'dividend':dividend,
        'div_yield':div_yield,
        'update_time':update_time
    }
    return tstock_info

# 抓取yahoo_fin股價
def get_stock_data(stock_no, start_dt, end_dt):
    data = get_data(stock_no , start_date = start_dt, end_date=end_dt)
    data['date'] = data.index
    data.dropna(subset=["close"], inplace=True)
    data = data.reset_index(drop=True)
    return data

# 建立回歸模型並顯示詳細內容
def detailed_linear_regression(X, y):
    reg = LinearRegression()
    reg.fit(X, y)
    # Store the parameters (regression intercept and coefficients) and predictions
    params = np.append(reg.intercept_, reg.coef_)
    predictions = reg.predict(X)
    # Create matrix with shape (num_samples, num_features + 1)
    # Where the first column is all 1s and then there is one column for the values
    # of each feature/predictor
    X_mat = np.append(np.ones((X.shape[0], 1)), X, axis = 1)
    # Compute residual sum of squares
    RSS = np.sum((y - predictions)**2)
    # Compute total sum of squares
    TSS = np.sum((np.mean(y) - y)**2)
    # Estimate the variance of the y-values
    obs_var = RSS/(X_mat.shape[0] - X_mat.shape[1])
    # Residual standard error is square root of variance of y-values
    RSE = obs_var**0.5
    # Variances of the parameter estimates are on the diagonal of the 
    # variance-covariance matrix of the parameter estimates
    var_beta = obs_var*(np.linalg.inv(np.matmul(X_mat.T, X_mat)).diagonal())
    # Standard error is square root of variance
    se_beta = np.sqrt(var_beta)
    # t-statistic for beta_i is beta_i/se_i, 
    # where se_i is the standard error for beta_i
    t_stats_beta = params/se_beta
    # Compute p-values for each parameter using a t-distribution with
    # (num_samples - 1) degrees of freedom
    beta_p_values = [2 * (1 - stats.t.cdf(np.abs(t_i), X_mat.shape[0] - 1))
                    for t_i in t_stats_beta]
    # Compute value of overall F-statistic, to measure how likely our
    # coefficient estimate are, assuming there is no relationship between
    # the predictors and the response
    F_overall = ((TSS - RSS)/(X_mat.shape[1] - 1))/(RSS/(X_mat.shape[0] - X_mat.shape[1]))
    F_p_value = stats.f.sf(F_overall, X_mat.shape[1] - 1, X_mat.shape[0] - X_mat.shape[1])
    # Construct dataframe for the overall model statistics:
    # RSE, R^2, F-statistic, p-value for F-statistic
    oa_model_stats = pd.Series({"Residual standard error": RSE, "R-squared": reg.score(X, y),
                                "F-statistic": F_overall, "F-test p-value": F_p_value})
    # Construct dataframe for parameter statistics:
    # coefficients, standard errors, t-statistic, p-values for t-statistics
    param_stats = pd.DataFrame({"Coefficient": params, "Standard Error": se_beta,
                                "t-value": t_stats_beta, "Prob(>|t|)": beta_p_values})
    return {"model": reg, "param_stats": param_stats, "oa_stats": oa_model_stats}

# 建立五線譜資料表
def get_fiveline_table(stock_history, prob1=68.26, prob2=95.44):
    s1 = (1-prob2/100)/2
    s2 = (1-prob1/100)/2
    s3 = 1-s2
    s4 = 1-s1
    x = (np.arange(len(stock_history))+1).reshape((-1, 1))
    y = np.array(stock_history.close)
    stock_reg = detailed_linear_regression(x,y)
    reg_line = stock_reg['model'].predict(x)
    Sd = stock_reg['oa_stats'][0]
    ss1 = norm.ppf(s1,reg_line,Sd)
    ss2 = norm.ppf(s2,reg_line,Sd)
    ss3 = norm.ppf(s3,reg_line,Sd)
    ss4 = norm.ppf(s4,reg_line,Sd)
    Rsquared = stock_reg['oa_stats'][1]
    Slope = stock_reg['model'].coef_[0]
    
    fiveline_data = pd.DataFrame({'stock_date':stock_history.date,
                              'stock_price':stock_history.close,
                              'reg-2sd':ss1,
                              'reg-1sd':ss2,
                              'reg_line':reg_line,
                              'reg+1sd':ss3,
                              'reg+2sd':ss4,
                              'rsquared':Rsquared,
                              'slope':Slope,
                              'sd':Sd
                              })
    fiveline_data = fiveline_data.reset_index(drop=True)
    return fiveline_data

# 根據五線譜資料衡量股價
def assess_status(fiveline_vec):
    intervals = np.array(fiveline_vec)[0][2:7]
    val = intervals > fiveline_vec.stock_price.values
    if(sum(val)==0): status = "股價高於歷史股價2倍標準差之外，絕對高點，建議看空。"
    if(sum(val)==1): status = "股價高於歷史股價1倍標準差之外，相對高點，建議持續觀察或看空。"
    if(sum(val)==2): status = "股價位於一般波動區間，且相對高點，建議持續觀察。"
    if(sum(val)==3): status = "股價位於一般波動區間，且相對低點，建議持續觀察。"
    if(sum(val)==4): status = "股價低於歷史股價1倍標準差之外，相對低點，建議持續觀察或看多。"
    if(sum(val)==5): status = "股價低於歷史股價2倍標準差之外，絕對低點，建議看多。"
    return status,sum(val)

# 五線譜畫圖
def plot_five_line(five_table):
    x = five_table['stock_date'].astype(str)
    stock_price = round(five_table['stock_price'],2)
    reg_m2sd = round(five_table['reg-2sd'],2)
    reg_m1sd = round(five_table['reg-1sd'],2)
    reg_line = round(five_table['reg_line'],2)
    reg_p1sd = round(five_table['reg+1sd'],2)
    reg_p2sd = round(five_table['reg+2sd'],2)
    line = Line()
    line.load_javascript()
    line.add_xaxis(xaxis_data=x)
    line.add_yaxis(
        series_name="price",
        label_opts=opts.LabelOpts(is_show=False),
        itemstyle_opts=opts.ItemStyleOpts(color='black'), 
        y_axis=stock_price,
        markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_='max'),opts.MarkPointItem(type_='min')])
        )
    line.add_yaxis(series_name="-2sd",label_opts=opts.LabelOpts(is_show=False), y_axis=reg_m2sd,color="#00BBFF")
    line.add_yaxis(series_name="-1sd",label_opts=opts.LabelOpts(is_show=False), y_axis=reg_m1sd,color="#00DD77")
    line.add_yaxis(series_name="reg line",label_opts=opts.LabelOpts(is_show=False), y_axis=reg_line,color="#888888")
    line.add_yaxis(series_name="+1sd",label_opts=opts.LabelOpts(is_show=False), y_axis=reg_p1sd,color="#00DD77")
    line.add_yaxis(series_name="+2sd",label_opts=opts.LabelOpts(is_show=False), y_axis=reg_p2sd,color="#00BBFF")
    line.set_global_opts(
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=True),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=3)
            ),
            ),
            datazoom_opts=opts.DataZoomOpts(type_="inside",range_start=0,range_end=100),
        )
    return Markup(line.dump_options())

# MA線
def calculate_ma(day_count: int, data):
    result: List[Union[float, str]] = []
    for i in range(len(data)):
        if i < day_count:
            result.append("-")
            continue
        sum_total = 0.0
        for j in range(day_count):
            sum_total += float(data[i - j])
        result.append(abs(float("%.3f" % (sum_total / day_count))))
    return result

# 股票燭台圖
def plot_stock_kline(data):
    # 轉換日期為文字
    data['date'] = data['date'].astype(str)
    kline = (
    Kline()
    .add_xaxis(list(data['date']))
    .add_yaxis(
        "K line", 
        data[['open', 'close', 'low', 'high', 'volume']].round(2).values.tolist(),
        itemstyle_opts=opts.ItemStyleOpts(color="#ec0000", color0="#00da3c")
    )
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(is_scale=True),
        yaxis_opts=opts.AxisOpts(
            is_scale=True,
            splitarea_opts=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
        ),
        datazoom_opts=[opts.DataZoomOpts(type_="inside",range_start=70,range_end=100)],
        )  
    )
    close = list(data['close'].round(2))
    line = (
        Line()
        .add_xaxis(xaxis_data=list(data['date']))
        .add_yaxis(
                series_name="MA5",
                y_axis=calculate_ma(day_count=5, data=close),
                is_smooth=True,
                is_symbol_show=False,
                is_hover_animation=False,
                linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(color="#984B4B")
            )
            .add_yaxis(
                series_name="MA10",
                y_axis=calculate_ma(day_count=10, data=close),
                is_smooth=True,
                is_symbol_show=False,
                is_hover_animation=False,
                linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(color="#64A600")
            )
            .add_yaxis(
                series_name="MA20",
                y_axis=calculate_ma(day_count=20, data=close),
                is_smooth=True,
                is_symbol_show=False,
                is_hover_animation=False,
                linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(color="#4F9D9D")
            )
            .add_yaxis(
                series_name="MA30",
                y_axis=calculate_ma(day_count=30, data=close),
                is_smooth=True,
                is_symbol_show=False,
                is_hover_animation=False,
                linestyle_opts=opts.LineStyleOpts(width=1, opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.LineStyleOpts(color="#7373B9")
            )
    )
    kline.overlap(line)
    return Markup(kline.dump_options())
    

# 呈現五線譜資訊頁面
def five_line_view(stock_no, start_dt, end_dt):
    if stock_no[-3:]=='.TW':
        tstock_id = stock_no.split('.')[0]
    else:
        tstock_id = ""
    tstock_info = get_tstock_info(tstock_id)

    try:
        # 抓股價
        stock_price = get_stock_data(stock_no, start_dt, end_dt)
        # 五線譜表
        five_table = get_fiveline_table(stock_price)
        # 衡量現況
        fiveline_vec=five_table[-1:]
        stock_status = assess_status(fiveline_vec)
        # 五線譜圖物件
        five_plot_obj = plot_five_line(five_table)
        kline_plot_obj =  plot_stock_kline(stock_price)

        stock_info = {
            'return_code':"",
            'stock_no':stock_no,
            'end_dt':end_dt,
            'start_dt':start_dt,
            'r2':round(fiveline_vec['rsquared'],4).values[0],
            'slope':round(fiveline_vec['slope'],4).values[0],
            'sd':round(fiveline_vec['sd'],4).values[0],
            'stock_price':round(fiveline_vec['stock_price'],2).values[0],
            'reg_m2sd':round(fiveline_vec['reg-2sd'],2).values[0],
            'reg_m1sd':round(fiveline_vec['reg-1sd'],2).values[0],
            'reg_line':round(fiveline_vec['reg_line'],2).values[0],
            'reg_p1sd':round(fiveline_vec['reg+1sd'],2).values[0],
            'reg_p2sd':round(fiveline_vec['reg+2sd'],2).values[0],
            'stock_status':stock_status[0]
        }
    except:
        stock_info = {
            'return_code':'查無資訊或輸入有誤',
            'stock_no':'',
            'end_dt':'',
            'start_dt':'',
            'r2':'',
            'slope':'',
            'sd':'',
            'stock_price':'',
            'reg_m2sd':'',
            'reg_m1sd':'',
            'reg_line':'',
            'reg_p1sd':'',
            'reg_p2sd':'',
            'stock_status':''
        }
        five_plot_obj = None
        kline_plot_obj = None

    return render_template('five_line.html', tstock_id=tstock_id, stock_info=stock_info, tstock_info=tstock_info, five_plot_obj=five_plot_obj, kline_plot_obj=kline_plot_obj)
    

