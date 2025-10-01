

"""
@author: Naveen N G
@date: 1-10-2025
@description: Data analysis using graph for the data cleanup preparation.
"""



import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dotenv import load_dotenv
from hugging_face_models.Login import login_hf
from datasets import load_dataset, Dataset, DatasetDict
import matplotlib.pyplot as plt

load_dotenv()
login_hf()

from data_curate import DataCurate
from upload_dataset import UploadDataset

""" Data finetunning techniques: 
    1) Investigate
    2) Data clean/ Parse
    3) Visualise
    4) Data quality
    5) Curate
    6) Save
"""


class DataVisualize:


    def data_investigate(self, dataset):
        datapoint = dataset[2]
        print(datapoint["title"])
        print(datapoint["description"])
        print(datapoint["features"])
        print(datapoint["details"])
        print(datapoint["price"])

        # How many have prices?
        prices = 0
        for datapoint in dataset:
            try:
                price = float(datapoint["price"])
                if price > 0:
                    prices += 1
            except ValueError as e:
                pass
        print(f"There are {prices:,} with prices which is {prices/len(dataset)*100:,.1f}%")

        # For those with prices, gather the price and the length
        prices = []
        lengths = []
        for datapoint in dataset:
            try:
                price = float(datapoint["price"])
                if price > 0:
                    prices.append(price)
                    contents = datapoint["title"] + str(datapoint["description"]) + str(datapoint["features"]) + str(datapoint["details"])
                    lengths.append(len(contents))
            except ValueError as e:
                pass

        return [lengths, prices]


    def data_visualise(self, content_length, prices):
        # Plot the distribution of lengths
        plt.figure(figsize=(15, 6))
        plt.title(f"Lengths: Avg {sum(content_length)/len(content_length):,.0f} and highest {max(content_length):,}\n")
        plt.xlabel('Length (chars)')
        plt.ylabel('Count')
        plt.hist(content_length, rwidth=0.7, color="lightblue", bins=range(0, 6000, 100))
        plt.show()

        # Plot the distribution of prices
        plt.figure(figsize=(15, 6))
        plt.title(f"Prices: Avg {sum(prices)/len(prices):,.2f} and highest {max(prices):,}\n")
        plt.xlabel('Price ($)')
        plt.ylabel('Count')
        plt.hist(prices, rwidth=0.7, color="orange", bins=range(0, 1000, 10))
        plt.show()



# def data_investigate(dataset):
#     dataVisualize = DataVisualize()
#     lengths, prices = dataVisualize.data_investigate(dataset)
#     dataVisualize.data_visualise(lengths, prices)

def data_curate():
    dDataCurate = DataCurate()
    items = dDataCurate.get_items_with_price()
    sample = items
    tokens = [item.token_count for item in items]
    prices = [item.price for item in items]
    # dataVisualize = DataVisualize()
    # dataVisualize.data_visualise(tokens, prices)
    upload_dataset(sample)

def upload_dataset(items):
        UploadDataset(items).upload()

if __name__ == "__main__":
    data_curate()
    


# Use this command to run : python src/model_training/product_price_estimate/data_visualize.py 