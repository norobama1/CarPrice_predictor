from flask import Flask,render_template,request,redirect
import pandas as pd
import numpy as np
import pickle
from sklearn.exceptions import InconsistentVersionWarning


app = Flask(__name__)
model=pickle.load(open('LinearRegression.pkl','rb'))
car = pd.read_csv('Cleaned Car.csv')


@app.route('/',methods=['GET','POST'])
def index():

    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique())
    fuel_type = car['fuel_type'].unique()

    companies.insert(0,'Select Company')
    return render_template('index.html',companies=companies,car_models=car_models,years=years,fuel_type=fuel_type)

@app.route('/predict',methods=['POST','GET'])
def predict():
        company = request.form['company']
        car_model = request.form['car_models']
        year = int(request.form['year'])
        fuel_type = request.form['fuel_type']
        kms_driven = int(request.form['kms_driven'])

        prediction=model.predict(pd.DataFrame([[car_model,company,year,kms_driven,fuel_type]],columns=['name','company','year','kms_driven','fuel_type']))
        return str(np.round(prediction[0],2))




if __name__ == "__main__":
    app.run(debug=True)