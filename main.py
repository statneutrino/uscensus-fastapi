from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from ml import inference as infer
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc remote add -d -f s3remote s3://census-fastapi")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
    

app = FastAPI()


class NewData(BaseModel):
    age: int
    workclass: str 
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            'example':
                {
                    'age': 20,
                    'workclass': 'Self-emp-not-inc',
                    'fnlgt': 205100,
                    'education': 'HS-grad',
                    'education_num': 9,
                    'marital_status': 'Married-civ-spouse',
                    'occupation': 'Exec-managerial',
                    'relationship': 'Wife',
                    'race': 'White',
                    'sex': 'Female',
                    'capital_gain': 0,
                    'capital_loss': 0,
                    'hours_per_week': 40,
                    'native_country': 'United-States',
                }
        }


@app.get("/")
async def read_root():
    return {"message": "Welcome to statneutrino's API. This is a greeting only."}


@app.post('/prediction')
async def predict_salary(input_data: NewData):

    input_dict = {key: [value] for key, value in input_data.dict().items()}
    df = pd.DataFrame.from_dict(input_dict)

    # Change underscores in df column names
    df.columns = df.columns.str.replace("_", "-")

    y_pred = infer.inference(df, output="string")
    return {'results': y_pred}

