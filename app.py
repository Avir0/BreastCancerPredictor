from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature_str = request.form['feature']
    feature_list = [float(x) for x in feature_str.split(',')]
    feature_array = np.array(feature_list).reshape(1, -1)

    # Predict
    pred = model.predict(feature_array)[0]
    output = "cancrous" if pred == 1 else "non-cancrous"

    return render_template('index.html', message=output)

if __name__ == '__main__':
    app.run(debug=True)
