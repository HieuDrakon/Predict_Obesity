# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Dummy Classifier and SVC models
dummy_filename = 'data_pkl/dummy_classifier.pkl'
svc_filename = 'data_pkl/svc_classifier.pkl'

dummy_model = pickle.load(open(dummy_filename, 'rb'))
svc_model = pickle.load(open(svc_filename, 'rb'))

# Define Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', prediction=-1)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data from the user input
        algorithm = request.form['algorithm']
        sex = request.form['sex']
        age = int(request.form['age'])
        height = float(request.form['height'])
        overweight_obese_family = int(request.form['overweight_obese_family'])
        fast_food_consumption = int(request.form['consumption_of_fast_food'])
        vegetable_consumption = int(request.form['frequency_of_consuming_vegetables'])
        main_meals = int(request.form['number_of_main_meals_daily'])
        food_between_meals = int(request.form['food_intake_between_meals'])
        smoking = int(request.form['smoking'])
        liquid_intake = float(request.form['liquid_intake_daily'])
        calorie_intake = float(request.form['calculation_of_calorie_intake'])
        physical_exercise = int(request.form['physical_exercise'])
        technology_schedule = int(request.form['schedule_dedicated_to_technology'])
        transportation_type = int(request.form['type_of_transportation_used'])

        # Prepare data for prediction
        data = np.array([[sex, age, height, overweight_obese_family, fast_food_consumption,
                          vegetable_consumption, main_meals, food_between_meals, smoking, 
                          liquid_intake, calorie_intake, physical_exercise, technology_schedule, 
                          transportation_type]])

        # Predict using both models
        dummy_prediction = dummy_model.predict(data)[0]
        svc_prediction = svc_model.predict(data)[0]
        # print (dummy_prediction)
        # print (svc_prediction)
        if algorithm == '1':
            prd = dummy_prediction
        if algorithm == '2':
            prd = svc_prediction 
        return render_template('result.html', prediction=prd)

if __name__ == '__main__':
    app.run(debug=True)
