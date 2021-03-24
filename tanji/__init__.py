from flask import Flask
from tanji.route import index,tables,user,login,register,password,p401,p404,p500,charts,logout,stock
from datetime import timedelta



def create_app():
    app=Flask(__name__)

    # login_manager = LoginManager()
    # login_manager.init_app(app)
    # login_manager.session_protection = "strong"
    # login_manager.login_view = 'login'
    # login_manager.login_message = '請先登入'

    app.secret_key = '063188051cc5effeb9d962713825fe01'
    app.permanent_session_lifetime = timedelta(minutes=10)
    app.add_url_rule('/','index',index)
    app.add_url_rule('/index','index',index)
    app.add_url_rule('/tables','tables',tables)
    app.add_url_rule('/user','user',user)
    app.add_url_rule('/login','login',login, methods=["POST","GET"])
    app.add_url_rule('/logout','logout',logout)
    app.add_url_rule('/register','register',register)
    app.add_url_rule('/password','password',password)
    app.add_url_rule('/401','401',p401)
    app.add_url_rule('/404','404',p404)
    app.add_url_rule('/500','500',p500)
    app.add_url_rule('/charts','charts',charts)
    app.add_url_rule('/stock','stock',stock, methods=["POST","GET"])
    return app