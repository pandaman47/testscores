import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app = Flask(__name__)

## Route for rendering the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  ## directing to home page where user can input data
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity= request.form.get('ethnicity'),
            parental_level_of_education= request.form.get('parental_level_of_education'),
            lunch= request.form.get('lunch'),
            test_preparation_course= request.form.get('test_preparation_course'),
            reading_score= float(request.form.get('reading_score')),
            writing_score= float(request.form.get('writing_score'))
        )
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        

        Predict_Pipeline = PredictPipeline()
        results = Predict_Pipeline.predict(pred_df)
        print(results)
        return render_template('home.html', results=results[0]) ### rendering the home page with the prediction result =, 
        #results will be in the form of a list, we are taking the first element of the list to display it on the home page


    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)  # Running the Flask app on port 5000
