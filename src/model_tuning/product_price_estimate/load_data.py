import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import pickle
from data_clean import DataCleanItem

current_dir = os.path.dirname(os.path.abspath(__file__))

class DataLoader:

    def __init__(self):
        pass

    def get_saved_data(self):
        # Let's avoid curating all our data again! Load in the pickle files:
        with open(f'{current_dir}/train.pkl', 'rb') as file:
            train = pickle.load(file)

        with open(f'{current_dir}/test.pkl', 'rb') as file:
            test = pickle.load(file)
        
        return [train, test]