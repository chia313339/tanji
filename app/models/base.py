from flask import render_template, redirect, url_for, request, send_from_directory, flash, jsonify
from flask import send_from_directory
import pandas as pd
import json
import psycopg2
import sqlalchemy
import datetime

# 正式環境DB
pgdb_config={
'host':'35.194.216.83',
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

