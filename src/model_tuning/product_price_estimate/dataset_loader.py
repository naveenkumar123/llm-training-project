
"""
@author: Naveen N G
@date: 1-10-2025
@description: Load the different dataset form from huggingface McAuley-Lab/Amazon-Reviews-2023.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dotenv import load_dotenv
from hugging_face_models.Login import login_hf
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


load_dotenv()
login_hf()

from data_clean import DataCleanItem
CHUNK_SIZE = 1000
MIN_PRICE = 0.5
MAX_PRICE = 999.49

class DatasetLoader:


    def __init__(self, name):
        self.name = name
        self.dataset = None

    def load(self, workers=8):
        """
        Load in this dataset; the workers parameter specifies how many processes
        should work on loading and scrubbing the data
        """
        start = datetime.now()
        print(f"Loading dataset {self.name}", flush=True)
        self.dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{self.name}", split="full", trust_remote_code=True)
        # self.dataset  = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_Appliances", split="full", trust_remote_code=True)

        results = self.load_in_parallel(workers)
        finish = datetime.now()
        print(f"Completed {self.name} with {len(results):,} datapoints in {(finish-start).total_seconds()/60:.1f} mins", flush=True)
        return results
    
    def load_in_parallel(self, workers):
        """
        Use concurrent.futures to farm out the work to process chunks of datapoints -
        This speeds up processing significantly, but will tie up your computer while it's doing so!
        """
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(pool.map(self.from_chunk, self.chunk_generator()), total=chunk_count):
                print('Inside the batch: ' + self.name)
                results.extend(batch)
        for result in results:
            result.category = self.name
        return results
    
    def chunk_generator(self):
        """
        Iterate over the Dataset, yielding chunks of datapoints at a time
        """
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))


    def from_chunk(self, chunk):
        """
        Create a list of Items from this chunk of elements from the Dataset
        """
        batch = []
        for datapoint in chunk:
            result = self.from_datapoint(datapoint)
            if result:
                batch.append(result)
        return batch
    

    def from_datapoint(self, datapoint):
        """
        Try to create an Item from this datapoint
        Return the Item if successful, or None if it shouldn't be included
        """
        try:
            price_str = datapoint['price']
            if price_str:
                price = float(price_str)
                if MIN_PRICE <= price <= MAX_PRICE:
                    item = DataCleanItem(datapoint, price)
                    return item if item.include else None
        except ValueError:
            return None