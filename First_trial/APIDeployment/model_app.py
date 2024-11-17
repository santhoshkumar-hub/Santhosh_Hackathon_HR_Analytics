# importing libraires
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib
import sklearn

app = FastAPI()

KPIs_met = 'KPIs_met >80%'
awards_won = 'awards_won?' 

class Input(BaseModel):
   department             : object
   region                 : object
   education              : object
   gender                 : object
   recruitment_channel    : object
   no_of_trainings        : int
   age                    : int
   previous_year_rating   : float
   length_of_service      : int
   KPIs_met               : int
   awards_won             : int
   avg_training_score     : int


class Output(BaseModel):
    is_promoted: int


@app.post("/predict")
def predict(data: Input) -> Output:
    X_input = pd.DataFrame([{'department': data.department,'region': data.region,'education': data.education,'gender':data.gender,
              'recruitment_channel' : data.recruitment_channel,'no_of_trainings' : data.no_of_trainings,'age': data.age,'previous_year_rating':data.previous_year_rating,
              'length_of_service' : data.length_of_service,'KPIs_met >80%': data.KPIs_met,'awards_won?' :data.awards_won,'avg_training_score': data.avg_training_score}])

 #   X_input.columns = ['department','region','education','gender','recruitment_channel',
 #                      'no_of_trainings','age','previous_year_rating','length_of_service','KPIs_met >80%',
 #                     'awards_won?','avg_training_score']

    #load model
    model = joblib.load('HR_promotion_Analytics_pipeline_model.pkl')

    #predict
    prediction = model.predict(X_input)

    #output
    return Output(is_promoted = prediction)





