

"""
@author: Naveen N G
@date: 1-10-2025
@description: Datacurate component.
"""


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dataset_loader import DatasetLoader





class DataCurate:


    # Create an CleanedItem object for each with a price
    def get_items_with_price(self):
        dataset_names = [
            # "Automotive",
            # "Electronics",
            # "Office_Products",
            # "Tools_and_Home_Improvement",
            # "Cell_Phones_and_Accessories",
            # "Musical_Instruments",
            "Toys_and_Games",
            "Appliances"

        ]   
        items = []
        for dataset_name in dataset_names:
            print(f'Dataset Loading for : {dataset_name} initiated')
            loader = DatasetLoader(dataset_name)
            dataset = loader.load(workers=6)
            print(f'Dataset Loading for : {dataset_name} completed')
            items.extend(dataset)

        return items