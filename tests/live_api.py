import requests
import pandas as pd


test_dict = {
    'age': [18, 50],
    'workclass': ['Never-worked', 'Self-emp-inc'],
    'fnlgt': [189778, 189778],
    'education': ['Preschool', 'Masters'],
    'education_num': [2, 15],
    'marital_status': ['Never-married', 'Married-AF-spouse'],
    'occupation': ['Priv-house-serv', 'Prof-specialty'],
    'relationship': ['Own-child', 'Wife'],
    'race': ['Amer-Indian-Eskimo', 'White'],
    'sex': ['Female', 'Male'],
    'capital_gain': [0, 60000],
    'capital_loss': [0, 0],
    'hours_per_week': [3, 50],
    'native_country': ['Mexico', 'Japan']
}
test_df = pd.DataFrame(test_dict)
test_json = test_df.head(1).to_dict(orient="records")[0]

response = requests.post(
    'https://uscensus-fastapi.herokuapp.com/prediction',
    json=test_json,
    headers={'Content-Type': 'application/json'}
)

print(response.status_code)
print(response.json())
