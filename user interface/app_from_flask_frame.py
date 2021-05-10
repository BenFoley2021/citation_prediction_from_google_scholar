"""


"""


from flask import Flask, render_template, request
from jinja2 import Template
from bokeh.resources import CDN
from bokeh.embed import json_item
from bokeh.plotting import figure
import json

import os
import pickle

app = Flask(__name__)
app.config.from_object(__name__)
    


@app.route('/')
def welcome():

    return render_template('form.html')


@app.route('/result', methods=['POST'])
def result():
    
    global model
    global dictWord2Vec
    ticker = request.form.get("var_1", type=str)

    operation = request.form.get("operation")
    
    data = get_data()
    data = num_to_shape_class(data)
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
                         
    return render_template('result.html', recipe_list = data)


def get_data():
    
    cwd = os.getcwd()
    return pickle.load(open(cwd + '\\data\\output_to_ui.pickle', 'rb'))

def num_to_shape_class(data):
    """ Converts the number in the rank to the class used in the html rendering
        currenlty the classes are circle1, circle2, etc
    """
    for thing in data:
        thing['title'] = '  ' + thing['title']
        thing['other_info'][0] = 'Journal:  ' + thing['other_info'][0]
        thing['other_info'][1] = 'Authors:  ' + thing['other_info'][1]
        if thing['rank'] > 5: # listing as class
            thing['rank'] = 'circle5'
        else:
            thing['rank'] = 'circle' + str(thing['rank'])
    
    return data

if __name__ == '__main__':

    #### setup vars, globals are bad and such, i know

    
    app.run(debug=False)

#             <fieldset disabled>