from worldbankapp import app
import json, plotly
from flask import render_template
from wrangling_scripts.wrangle_data import return_figures

# 入口文件 

@app.route('/')       # 路由， '/'表示访问路径的时候执行一下函数，返回字符串显示在web上
@app.route('/index')
def index():

    figures = return_figures()

    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html',
                           ids=ids,
                           figuresJSON=figuresJSON)