"""


"""


from flask import Flask, render_template, request, g, current_app

app = Flask(__name__)
app.config.from_object(__name__)

import pickle

    
@app.route('/')
def welcome():

    return render_template('form.html')


@app.route('/result', methods=['POST'])
def result():
    
    global model
    global dictWord2Vec
    
    title = request.form.get("var_1", type=str)
    year = request.form.get("var_2", type=str)
    jName = request.form.get("var_3", type=str)
    author = request.form.get("var_4", type=str)
    operation = request.form.get("operation")
    
    wordsForBow = {'title': title, 'otherFactor': year + ' ' + jName + ' ' + author}
    
    # if(operation == 'Predict'):
    #     result = testFunc(var_1,var_2)
    #     result = setUpPredict(wordsForBow)
    MLoutput, factUsed = mlObj.setUpPredict(wordsForBow)

    entry = MLoutput
    entry2 = factUsed
    return render_template('result.html', entry=entry, entry2 = entry2)







#mlObj = doML()  ####### initializing the object containing the model

if __name__ == '__main__':

    #### setup vars, globals are bad and such, i know

    
    app.run(debug=True)

#             <fieldset disabled>