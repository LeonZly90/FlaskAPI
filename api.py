# Dependencies
import sys
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if reg:
        try:
            test_data = request.json
            test_data = list(test_data[0].values())
            test_dataframe = pd.DataFrame(columns=['STAGE_CODE', 'MARKET_TYPE'])
            for index, column in enumerate(test_dataframe.columns):
                test_dataframe[column] = [test_data[index]]

            feature = model_standardscaler.transform(np.array(test_data[-1]).reshape(-1, 1))
            one_hot = model_ohe.transform(test_dataframe).toarray()
            test = np.concatenate([one_hot, feature], axis=1)
            prediction = reg.predict(test)
            result = str(round(prediction[0], 2))
            return 'For new data forecast:' + result + ' hours.'
            # return jsonify({'prediction hours': str(prediction)})

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return 'No model here to use'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])  # This is for a command-line input
        print(predict())

    except:
        port = 33333  # If you don't provide any port the port will be set to 33333

    reg = joblib.load("model.pkl")  # Load "model.pkl"
    print('Model loaded')
    model_ohe = joblib.load('model_ohe.pkl')
    print('model_ohe loaded')
    model_standardscaler = joblib.load('model_standardscaler.pkl')
    print('model_standardscaler loaded')
    app.run(port=port, debug=True)
    # ['BO', 'Higher Education', 30000000]
