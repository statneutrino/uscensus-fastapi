from fastapi.testclient import TestClient
import pandas as pd
import pytest
from main import app

client = TestClient(app)


@pytest.fixture
def test_df():
    """
    Pandas df with two test cases for inferences
    First row should return '<=50K' or 0
    Second row should return '>50K' or 1
    """
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
    return test_df


# Test GET on root
def test_root_status():
    r = client.get("/")
    assert r.status_code == 200


def test_root_request_contents():
    r = client.get("/")
    assert r.json() == {
        "message": "Welcome to statneutrino's API. This is a greeting only."}


# Test POST method with test data
def test_post_low_salary(test_df):
    test_json = test_df.head(1).to_dict(orient="records")[0]
    r = client.post(
        '/prediction',
        json=test_json,
        headers={'Content-Type': 'application/json'}
    )
    print(r.json())
    assert r.status_code == 200
    assert r.json() == {'results': '<=50K'}


def test_post_high_salary(test_df):
    test_json = test_df.tail(1).to_dict(orient="records")[0]
    print(test_json)
    r = client.post(
        '/prediction',
        json=test_json,
        headers={'Content-Type': 'application/json'}
    )

    assert r.status_code == 200
    assert r.json() == {'results': '>50K'}
