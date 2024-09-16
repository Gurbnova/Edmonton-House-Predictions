import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, bedroom, bath, sqft):
    try:
        loc_index = __data_columns.index(location.lower())  # finding the column index of the wanted location lowercase bcoz json is lowercase
    except:
        #print('NOT IN THE LIST')
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = bedroom
    x[1] = bath
    x[2] = sqft
    if loc_index >= 0:
        x[loc_index] = 1
    
    x_df = pd.DataFrame([x], columns=[ column.title() for column in __data_columns])

    return round(__model.predict(x_df)[0], 2)

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model
    base_path = Path(__file__).resolve().parent

    columns_path = base_path / 'artifacts' / 'columns.json'
    with open(columns_path, "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3 : ]

    model_path = base_path / 'artifacts' / 'edmonton_home_prices_model.pickle'
    with open(model_path, "rb") as f:
        __model = pickle.load(f)
    print("loading the artifacts...done")

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('queen mary park', 2, 2, 1000))
    print(get_estimated_price('queen mary park', 3, 3, 1000))
    print(get_estimated_price('allendale', 2, 2, 1000)) # other location
    print(get_estimated_price('jackson heights', 2, 2, 1000)) # other location
    

