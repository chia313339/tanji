from flask import Flask
from app.route import *

def create_app():
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    app.config['SECRET_KEY'] = 'ohthisisatanjiwebapp12345'
    app.add_url_rule('/', 'index', index, methods=['GET', 'POST'])
    app.add_url_rule('/index', 'index', index, methods=['GET', 'POST'])
    app.add_url_rule('/ml_stock', 'ml_stock', ml_stock, methods=['GET', 'POST'])
    app.add_url_rule('/stock_list', 'stock_list', stock_list, methods=['GET', 'POST'])
    app.add_url_rule('/five_line', 'five_line', five_line, methods=['GET', 'POST'])
    app.add_url_rule('/get_stock_list_csv', 'get_stock_list_csv', get_stock_list_csv, methods=['GET', 'POST'])
    app.add_url_rule('/five_line_view/<stock_no>/<start_dt>/<end_dt>', 'five_line_view', five_line_view, methods=['GET','POST'])
    app.register_error_handler(404, page_not_found)
    return app