import pandas as pd
import numpy as np
import requests
from scipy.stats import norm
from scipy import stats
from sklearn.linear_model import LinearRegression
from datetime import datetime
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from yahoo_fin import stock_info as si
from yahoo_fin.stock_info import *

import psycopg2
import sqlalchemy
from sqlalchemy import create_engine

pgdb_config={
'host':'130.211.245.233',
'port':5432,
'user':'tanji',
'password':'tanji',
'database':'tanji',
}

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
    if(sum(val)==0): status = "2倍標準差之外絕對高點"
    if(sum(val)==1): status = "1倍標準差之外相對高點"
    if(sum(val)==2): status = "波動區間相對高點"
    if(sum(val)==3): status = "波動區間相對低點"
    if(sum(val)==4): status = "1倍標準差之外相對低點"
    if(sum(val)==5): status = "2倍標準差之外絕對低點"
    return status,sum(val)

# 計算股票狀態
def stock_list_five_line(df):
  stock_status_l1y = []
  stock_status_l2y = []
  stock_status_l3y = []
  
  end_dt = datetime.datetime.now().strftime("%Y-%m-%d")
  start_l1y = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
  start_l2y = (datetime.datetime.now() - datetime.timedelta(days=365*2)).strftime("%Y-%m-%d")
  start_l3y = (datetime.datetime.now() - datetime.timedelta(days=365*3)).strftime("%Y-%m-%d")
  
  for index,stock_no in enumerate(df['代號']+'.TW'):
    print("開始判讀第",index,"個，代號",stock_no)
    try:
      stock_price = get_stock_data(stock_no, start_l3y, end_dt)

      stock_price_tmp = stock_price[stock_price['date']>start_l1y]
      five_table_tmp = get_fiveline_table(stock_price_tmp)
      fiveline_vec_tmp=five_table_tmp[-1:]
      stock_status_l1y_tmp = assess_status(fiveline_vec_tmp)[0]

      stock_price_tmp = stock_price[stock_price['date']>start_l2y]
      five_table_tmp = get_fiveline_table(stock_price_tmp)
      fiveline_vec_tmp=five_table_tmp[-1:]
      stock_status_l2y_tmp = assess_status(fiveline_vec_tmp)[0]

      stock_price_tmp = stock_price[stock_price['date']>start_l3y]
      five_table_tmp = get_fiveline_table(stock_price_tmp)
      fiveline_vec_tmp=five_table_tmp[-1:]
      stock_status_l3y_tmp = assess_status(fiveline_vec_tmp)[0]

    except:
      stock_status_l1y_tmp = '無資料，無法觀察。'
      stock_status_l2y_tmp = '無資料，無法觀察。'
      stock_status_l3y_tmp = '無資料，無法觀察。'
      
    finally:
      stock_status_l1y.append(stock_status_l1y_tmp)
      stock_status_l2y.append(stock_status_l2y_tmp)
      stock_status_l3y.append(stock_status_l3y_tmp)
      
  df['five_line_1y'] = stock_status_l1y
  df['five_line_2y'] = stock_status_l2y
  df['five_line_3y'] = stock_status_l3y
  
  return df

# 計算美股狀態
def ustock_table(ustock_list):
  year = datetime.datetime.now().year
  user_agent = UserAgent()
  headers = {'user-agent': user_agent.random}
  stock_name = []
  price = []
  dividend = []
  yield_ = []
  stock_l1y_r2 = []
  stock_l1y_slope = []
  stock_l1y_sd = []

  stock_status_l1y = []
  stock_status_l2y = []
  stock_status_l3y = []

  end_dt = datetime.datetime.now().strftime("%Y-%m-%d")
  start_l1y = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
  start_l2y = (datetime.datetime.now() - datetime.timedelta(days=365*2)).strftime("%Y-%m-%d")
  start_l3y = (datetime.datetime.now() - datetime.timedelta(days=365*3)).strftime("%Y-%m-%d")

  for index,stock_no in enumerate(ustock_list):
    print("開始判讀第",index,"個，代號",stock_no)
    url = "http://www.aastocks.com/tc/usq/quote/quote.aspx?symbol=%s" % stock_no
    # 獲取 html 資訊
    try:
      stock_price = get_stock_data(stock_no, start_l3y, end_dt)

      stock_price_tmp = stock_price[stock_price['date']>start_l1y]
      five_table_tmp = get_fiveline_table(stock_price_tmp)
      fiveline_vec_tmp=five_table_tmp[-1:]
      stock_status_l1y_tmp = assess_status(fiveline_vec_tmp)[0]
      stock_l1y_r2_tmp = round(fiveline_vec_tmp['rsquared'].values[0],4)
      stock_l1y_slope_tmp = round(fiveline_vec_tmp['slope'].values[0],4)
      stock_l1y_sd_tmp = round(fiveline_vec_tmp['sd'].values[0],4)

      stock_price_tmp = stock_price[stock_price['date']>start_l2y]
      five_table_tmp = get_fiveline_table(stock_price_tmp)
      fiveline_vec_tmp=five_table_tmp[-1:]
      stock_status_l2y_tmp = assess_status(fiveline_vec_tmp)[0]

      stock_price_tmp = stock_price[stock_price['date']>start_l3y]
      five_table_tmp = get_fiveline_table(stock_price_tmp)
      fiveline_vec_tmp=five_table_tmp[-1:]
      stock_status_l3y_tmp = assess_status(fiveline_vec_tmp)[0]
    except:
      stock_status_l1y_tmp = '無資料，無法觀察。'
      stock_status_l2y_tmp = '無資料，無法觀察。'
      stock_status_l3y_tmp = '無資料，無法觀察。'
      stock_l1y_r2_tmp = None
      stock_l1y_slope_tmp = None
      stock_l1y_sd_tmp = None
    
    res = requests.get(url, headers = headers)
    res.encoding='utf-8'
    tmp = BeautifulSoup(res.text, 'lxml')
    # stock_name
    try:
      stock_name_tmp = tmp.select('#SQ_Name')[0].text
    except:
      stock_name_tmp = ""
    # price
    try:
      price_tmp = round(si.get_live_price(stock_no),2)
    except:
      price_tmp = 0
    # dividend
    try:
      dividend_tmp = round(sum(si.get_dividends(stock_no)[si.get_dividends(stock_no).index.year == year-1]['dividend']),2)
    except:
      dividend_tmp = 0
    # yield
    try:
      yield_tmp = round(dividend/price,4)
    except:
      yield_tmp = 0

    stock_status_l1y.append(stock_status_l1y_tmp)
    stock_status_l2y.append(stock_status_l2y_tmp)
    stock_status_l3y.append(stock_status_l3y_tmp)
    stock_l1y_r2.append(stock_l1y_r2_tmp)
    stock_l1y_slope.append(stock_l1y_r2_tmp)
    stock_l1y_sd.append(stock_l1y_r2_tmp)

    stock_name.append(stock_name_tmp)
    price.append(price_tmp)
    dividend.append(dividend_tmp)
    yield_.append(yield_tmp)

  return pd.DataFrame(list(zip( ustock_list, stock_name, price, dividend, yield_, stock_status_l1y, stock_l1y_r2, stock_l1y_slope, stock_l1y_sd,  stock_status_l2y, stock_status_l3y)), columns =['stock_no', 'stock_name', 'price', 'dividend', 'yield_', 'stock_status_l1y', 'stock_l1y_r2', 'stock_l1y_slope', 'stock_l1y_sd', 'stock_status_l2y', 'stock_status_l3y'])

# 計算美股狀態無爬網
def ustock_table_n(ustock_list):
  year = datetime.datetime.now().year
  stock_name = []
  price = []
  dividend = []
  yield_ = []
  stock_l1y_r2 = []
  stock_l1y_slope = []
  stock_l1y_sd = []

  stock_status_l1y = []
  stock_status_l2y = []
  stock_status_l3y = []

  end_dt = datetime.datetime.now().strftime("%Y-%m-%d")
  start_l1y = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
  start_l2y = (datetime.datetime.now() - datetime.timedelta(days=365*2)).strftime("%Y-%m-%d")
  start_l3y = (datetime.datetime.now() - datetime.timedelta(days=365*3)).strftime("%Y-%m-%d")

  for index,stock_no in enumerate(ustock_list):
    print("開始判讀第",index,"個，代號",stock_no)
    # 獲取 html 資訊
    try:
      stock_price = get_stock_data(stock_no, start_l3y, end_dt)

      stock_price_tmp = stock_price[stock_price['date']>start_l1y]
      five_table_tmp = get_fiveline_table(stock_price_tmp)
      fiveline_vec_tmp=five_table_tmp[-1:]
      stock_status_l1y_tmp = assess_status(fiveline_vec_tmp)[0]
      stock_l1y_r2_tmp = round(fiveline_vec_tmp['rsquared'].values[0],4)
      stock_l1y_slope_tmp = round(fiveline_vec_tmp['slope'].values[0],4)
      stock_l1y_sd_tmp = round(fiveline_vec_tmp['sd'].values[0],4)

      stock_price_tmp = stock_price[stock_price['date']>start_l2y]
      five_table_tmp = get_fiveline_table(stock_price_tmp)
      fiveline_vec_tmp=five_table_tmp[-1:]
      stock_status_l2y_tmp = assess_status(fiveline_vec_tmp)[0]

      stock_price_tmp = stock_price[stock_price['date']>start_l3y]
      five_table_tmp = get_fiveline_table(stock_price_tmp)
      fiveline_vec_tmp=five_table_tmp[-1:]
      stock_status_l3y_tmp = assess_status(fiveline_vec_tmp)[0]
    except:
      stock_status_l1y_tmp = '無資料，無法觀察。'
      stock_status_l2y_tmp = '無資料，無法觀察。'
      stock_status_l3y_tmp = '無資料，無法觀察。'
      stock_l1y_r2_tmp = None
      stock_l1y_slope_tmp = None
      stock_l1y_sd_tmp = None

    # stock_name
    try:
      stock_name_tmp = si.get_company_info(stock_no)['Value']['longBusinessSummary'].split('.')[0]
    except:
      stock_name_tmp = ""
    # price
    try:
      price_tmp = round(si.get_live_price(stock_no),2)
    except:
      price_tmp = 0
    # dividend
    try:
      dividend_tmp = round(sum(si.get_dividends(stock_no)[si.get_dividends(stock_no).index.year == year-1]['dividend']),2)
    except:
      dividend_tmp = 0
    # yield
    try:
      yield_tmp = round(dividend_tmp/price_tmp,4)
    except:
      yield_tmp = 0

    stock_status_l1y.append(stock_status_l1y_tmp)
    stock_status_l2y.append(stock_status_l2y_tmp)
    stock_status_l3y.append(stock_status_l3y_tmp)
    stock_l1y_r2.append(stock_l1y_r2_tmp)
    stock_l1y_slope.append(stock_l1y_r2_tmp)
    stock_l1y_sd.append(stock_l1y_r2_tmp)

    stock_name.append(stock_name_tmp)
    price.append(price_tmp)
    dividend.append(dividend_tmp)
    yield_.append(yield_tmp)

  return pd.DataFrame(list(zip( ustock_list, stock_name, price, dividend, yield_, stock_status_l1y, stock_l1y_r2, stock_l1y_slope, stock_l1y_sd,  stock_status_l2y, stock_status_l3y)), columns =['stock_no', 'stock_name', 'price', 'dividend', 'yield_', 'stock_status_l1y', 'stock_l1y_r2', 'stock_l1y_slope', 'stock_l1y_sd', 'stock_status_l2y', 'stock_status_l3y'])


print(datetime.datetime.now(),"排程開始")
# gather stock symbols from major US exchanges
df1 = pd.DataFrame( si.tickers_sp500() )
df2 = pd.DataFrame( si.tickers_nasdaq() )
df3 = pd.DataFrame( si.tickers_dow() )
df4 = pd.DataFrame( si.tickers_other() )

# convert DataFrame to list, then to sets
sym1 = set( symbol for symbol in df1[0].values.tolist() )
sym2 = set( symbol for symbol in df2[0].values.tolist() )
sym3 = set( symbol for symbol in df3[0].values.tolist() )
sym4 = set( symbol for symbol in df4[0].values.tolist() )

# join the 4 sets into one. Because it's a set, there will be no duplicate symbols
symbols = set.union( sym1, sym2, sym3, sym4 )

# Some stocks are 5 characters. Those stocks with the suffixes listed below are not of interest.
my_list = ['W', 'R', 'P', 'Q']
del_set = set()
sav_set = set()

print(datetime.datetime.now(),"抓取美股資料")

for symbol in symbols:
    if len( symbol ) > 4 and symbol[-1] in my_list:
        del_set.add( symbol )
    else:
        sav_set.add( symbol )


del_set = list(filter(lambda x: x != "", list(del_set)))
sav_set = list(filter(lambda x: x != "", list(sav_set)))

print( f'Removed {len( del_set )} unqualified stock symbols...' )
print( f'There are {len( sav_set )} qualified stock symbols...' )

print(datetime.datetime.now(),"美股資料抓取完成")
print(datetime.datetime.now(),"開始計算美股資料")

sav_set = list(set(sav_set))

step = 250
ustock_loop = [sav_set[i:i+step] for i in range(0,len(sav_set),step)]

engine = create_engine("postgresql://%s:%s@%s:5432/%s" % (pgdb_config['user'],pgdb_config['password'],pgdb_config['host'],pgdb_config['database']))


for index,tmp_list in enumerate(ustock_loop):
  print(datetime.datetime.now(),'第',index+1,'Partition 計算，有',len(tmp_list),'筆資料')
  df = ustock_table_n(tmp_list)
  df['update_time'] = datetime.datetime.now().strftime('%Y-%m-%d')
  print(datetime.datetime.now(),"資料表處理完成")
  print(datetime.datetime.now(),"寫入資料庫")
  if index==0:
    df.to_sql('ustock_table_tmp', engine, if_exists='replace',index=False,method='multi')
  else:
    df.to_sql('ustock_table_tmp', engine, if_exists='append',index=False,method='multi')

print(datetime.datetime.now(),"Table rename")

sqls = '''ALTER TABLE ustock_table
RENAME TO ustock_table_old;
ALTER TABLE ustock_table_tmp
RENAME TO ustock_table;
drop table ustock_table_old'''

conn = psycopg2.connect(**pgdb_config)
cursor = conn.cursor()
cursor.execute(sqls)
conn.commit()
conn.close()


print(datetime.datetime.now(),"排程完成")