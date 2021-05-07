# http://127.0.0.1:5000/


from flask import Flask, render_template, request, redirect
import os
import pandas as pd
import re
#root_dir = '/Users/aliceallafort/Google_Drive/Github/'
#save_dir = root_dir + 'Miamiam/data_save/'
app = Flask(__name__)
#img_dir = os.path.join('static', 'img')
#app.config['img_dir'] = img_dir
app.vars = {'ingredients_kw': ''}

@app.route('/index', methods=['GET', 'POST'])
def index():
    # if request.method == 'GET':
    #     return render_template('index.html')
    # else:
    #     app.vars['ingredients_kw'] = request.form['ingredients_kw']
    #     results={}
    #     results['ingredients_kw'] = app.vars['ingredients_kw']
    #     results['recipe_list'] = get_recipes(df, kw=app.vars['ingredients_kw'])
    
    # need title and ingredients
    list_of_dicts = [{'title': 'test_thing', 'ingredients': 'test_ingredient'}]

    return render_template('result checkpoint 5-6 635.html', **list_of_dicts)
    
# def get_recipes(df,rec_id=None, kw=None, n=3):
#     """
#     Query on recipe dataframe from keyword ingredients, matching all of them
#     :param ingredients_kw: List of ingredients
#     :param n: Number of recipes returned
#     :return: dictionary of the first N results
#     """
#     if kw is not None:
#         kw = [k.strip() for k in kw.split(' ')]
#         mask = df.ing_cleaned_all.apply(lambda t: match_string(kw, t, 'all'))
#     if rec_id is not None:
#         mask = df.recipe_id == int(rec_id)
#     return df[mask][['recipe_id', 'title', 'ing_cleaned', 'ingredients', 'directions']].to_dict('records')[:n]

if __name__ == "__main__":
    app.run(debug=True)  # DEBUGGING