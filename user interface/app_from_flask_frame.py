"""


"""


from flask import Flask, render_template, request
from jinja2 import Template
from bokeh.resources import CDN
from bokeh.embed import json_item
from bokeh.plotting import figure
import json
app = Flask(__name__)
app.config.from_object(__name__)
    
class storeTicker:
    def __init__(self):
        self.ticker = None
        
    def update_ticker(self,new_ticker):
        self.ticker = new_ticker

def get_template():
    landing_page = Template("""
    <!DOCTYPE html>
    <head>
      {{ resources }}
      <title>Example Visualization Web App</title>
    </head>
    <body>
      <div id="visualization">Example Visualization</div>
      <script>
        fetch('/visual')
          .then(function(response) {return response.json();})
          .then(function(item) {return Bokeh.embed.embed_item(item);})
      </script>
    </body>
    """)
    
    return landing_page


@app.route('/')
def welcome():

    return render_template('form.html')


@app.route('/result', methods=['POST'])
def result():
    
    global model
    global dictWord2Vec
    ticker = request.form.get("var_1", type=str)

    operation = request.form.get("operation")
    
    #ticker_storage.update_ticker(ticker)
    # if(operation == 'Predict'):
    #     result = testFunc(var_1,var_2)
    #     result = setUpPredict(wordsForBow)
    #p = produce_visual()
    #landing_page = get_template()
    #return render_template('result.html', entry=entry, entry2 = entry2)
    list_of_dicts = [{'title': 'test_thing', 'other_info': ['info_1', 'info_2'], 'rank': 'circle1'},
                     {'title': 'test_thing2', 'other_info': ['wasabi'], 'rank': 'circle1'},
                     {'title': 'test_thing3', 'other_info': ['steak'], 'rank': 'circle2'},
                     {'title': 'test_thing4', 'other_info': ['wings'], 'rank': 'circle3'}]
                         
    return render_template('result.html', recipe_list = list_of_dicts)

@app.route('/visual')
def produce_visual():
    import pandas as pd
    import requests
    
    def make_url(ticker):
        # example
        # https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=qqq&apikey=7TQJQIJPMNSM675B
        url_start = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="
        url_end = "&apikey="
        api_key = "7TQJQIJPMNSM675B"
        
        url = url_start + ticker + url_end + api_key
        return url
    
    def get_and_format_data(url):
        hist_prices = requests.get(url)

        hist_prices = hist_prices.json()
        
        daily_prices = hist_prices["Time Series (Daily)"]
        
        df = pd.DataFrame(data = daily_prices)
        
        df = df.transpose()
        
        df = df.astype(float)
        df['date'] = df.index
        df['date'] = pd.to_datetime(df['date'])
        #df['4. close'].plot()
    
        return df
    
    ticker = ticker_storage.ticker

    url = make_url(ticker)
    df = get_and_format_data(url)
    #fig = figure(plot_width=800, plot_height=250, x_axis_type="datetime")
    
    p = figure(plot_width=800, plot_height=250, x_axis_type="datetime")
    p.line(df['date'], df['4. close'], color='navy', alpha=0.5)

    #old way of displaying from tutorial
    #return json.dumps(json_item(p, "visualization"))

    # instead we just want to get the p object
    return json.dumps(json_item(p, "visualization"))

ticker_storage = storeTicker()

if __name__ == '__main__':

    #### setup vars, globals are bad and such, i know

    
    app.run(debug=False)

#             <fieldset disabled>