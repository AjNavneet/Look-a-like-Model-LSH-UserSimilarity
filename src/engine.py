##### Importing the required packages #####
import numpy as np
import pandas as pd
import pickle
from yaml import CLoader as Loader, load
from datasketch import MinHash, MinHashLSHForest
from ml_pipeline.utils import read_data_json, read_config, read_data_csv
from ml_pipeline.processing import list_to_integers, remove_rows, empty_values
from ml_pipeline.model import LSHGraph
from ml_pipeline.score import score_fn, get_extn
from ml_pipeline.utils import feat_imp, count_fn

##### Read the configuration file #####
config = read_config("../input/config.yaml")

##### Data Preprocessing - Data Cleaning  #####

# Read the JSON data file
data = read_data_json(config['data_path'])

# Reset the index in the data
data.reset_index(drop=True, inplace=True)

# Convert lists to integers
data = list_to_integers(features=config['features'],
                        list_cols=config['list_cols'],
                        data=data)

# Remove rows with values higher than the threshold (e.g., 16)
data = remove_rows(list_cols=config['list_cols'],
                  data=data)

# Replace empty values with an empty list
data = empty_values(list_cols=config['list_cols'],
                    data=data)

# Rename the index as 'id'
data.reset_index(inplace=True)
data.rename({"index": "id"}, axis=1, inplace=True)

# Write the clean data to a new JSON file
data.to_json(config['clean_data_path'])

# Calculate feature counts for scoring
count_df = count_fn(data,
                    features=config['features'],
                    list_cols=config['list_cols'])

count_df.to_csv(config["count_path"], index=False)

##### Model Training - Model Training  #####

# Read the cleaned data
data = read_data_json(config['clean_data_path'])
df = data.drop(config["label"], axis=1)

# Create a MinHashForest model
lsh = MinHashLSHForest(num_perm=config['n_perm'])

# Create an LSH graph object
lsh_graph = LSHGraph(df, lsh, features=config['features'],
                     id_col=config['id_col'],
                     n_perm=10)

# Train the model
lsh_graph.update_graph()

# Save the model to disk
with open(config['model_path'], "wb") as f:
    pickle.dump(lsh_graph, f)

##### Seed Set Extension #####

# Read the cleaned data
data = read_data_json(config['clean_data_path'])

# Read the seed set
seed = read_data_csv(config['seed_path'])
seed_ids = list(seed["id"])

# Load the trained model
lsh_graph = pickle.load(open(config['model_path'], "rb"))

# Retrieve the neighbors of the seed set from the LSH graph
neighbors = lsh_graph.extract_neighbors(seed_ids)

# Select records that are not in the seed set
df = data[~data["id"].isin(seed_ids)]

# Calculate the default click rate
label = config['label']
def_click_rate = df[label].mean()
def_click_rate = round(def_click_rate * 100, 2)

# Score the neighbors
df = score_fn(data,
             count_path=config['count_path'],
             features=config['features'],
             list_cols=config['list_cols'],
             seed_ids=list(seed["id"]),
             neighbors=lsh_graph.extract_neighbors(seed_ids),
             label=config["label"])

# Create and store the extension file
extn_click_rate = get_extn(df, seed_ids,
                          label=config["label"],
                          extn_path=config['extn_path'], x=2)

##### Analyze the Change in Click Rate #####

# Calculate and print the change in click rate
# An increase in click rate suggests that our model has performed well
# print(f"Click rate increased from {def_click_rate}% to {extn_click_rate}%")
