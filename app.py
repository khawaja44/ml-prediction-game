import logging
import sys

import numpy as np

from flask import Flask , request , jsonify , render_template
import gunicorn
import pickle

app = Flask (__name__, template_folder='Templates')
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

model = pickle.load(open('svc_trained_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html' ,results="")

@app.route('/output',methods=['POST'])
def predict():

    inputs = []

    tls = request.form.get('tls')
    tms = request.form.get('tms')
    trs = request.form.get('trs')

    mls = request.form.get('mls')
    mms = request.form.get('mms')
    mrs = request.form.get('mrs')

    bls = request.form.get('bls')
    bms = request.form.get('bms')
    brs = request.form.get('brs')

    results = {"tls":tls,"tms":tms,"trs":trs,"mls":mls,"mms":mms,"mrs":mrs,"bls":bls,"bms":bms,"brs":brs}    

    inputs.append(tls)
    inputs.append(tms)
    inputs.append(trs)

    inputs.append(mls)
    inputs.append(mms)
    inputs.append(mrs)

    inputs.append(bls)
    inputs.append(bms)
    inputs.append(brs)

    final_inputs = [np.array(inputs)]
    prediction = model.predict(final_inputs)


    if(prediction[0] == 1):
        return render_template('index.html',predicted_result = 'Player 1 Win',results = results)
    
    if(prediction[0] == 0):
        return render_template('index.html',predicted_result = 'Player 2 Win',results = results)


if __name__ == "__main__":
    app.run(debug = True)