from flask import Flask

app = Flask(__name__) 
# 传递参数为app的名称， __name__表示当前文件的名称

from worldbankapp import routes
