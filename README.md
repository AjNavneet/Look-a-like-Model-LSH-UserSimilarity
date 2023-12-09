# Lookalike Modeling using Locality Sensitive Hashing

## Business Objective

In today's digital age, the primary goal is to increase brand awareness and boost product sales. Online advertising is the most straightforward approach to achieve this. Online advertising is an effective way for businesses of all sizes to expand their reach, acquire new customers, and diversify revenue streams.

Online advertising is about using the internet as a powerful platform to deliver marketing messages to a specific audience. Social media, like Facebook, Instagram, Twitter, and more, has become a popular online pastime for people worldwide. Advertisers have adapted their strategies to target consumers on these platforms. This approach not only attracts website traffic but also enhances brand exposure, ultimately leading to increased sales. The central purpose of online advertising is to encourage potential customers to make a purchase.

However, not all individuals who see these ads will be interested in the product or service. The click rate is a measure that helps us determine the percentage of people who view an online ad and then click on it. The goal of online advertising is to reach the maximum number of relevant users.

To find these relevant users, we employ a method called lookalike modelling. Lookalike models are designed to build larger audiences based on the characteristics of a smaller group known as seed users. These seed users represent the ideal target audience.

In this project, our aim is to create a Lookalike model using the Locality Sensitive Hashing (LSH) algorithm to find similar and relevant customers. This model is expected to improve the click rate.

---

## Data Description

The dataset used in this project is provided by 'Adform' and originates from a specific online digital campaign. Given the dataset's vast size, we will work with a subset of it. The dataset is in JSON format and includes the following fields:

- "l": A binary label indicating whether the ad was clicked (1) or not (0).
- "c0" - "c9": Categorical features that have been hashed into 32-bit integers.
- Among these, "c6" and "c9" have multiple values per user, while the rest have single values per user.

For more information about the dataset, you can refer to this [link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TADBY7).

---

## Aim

Our goal is to build a lookalike model using the Locality Sensitive Hashing algorithm to find similar users and increase the click rate compared to the default rate.

---

## Tech Stack

- **Language**: `Python`
- **Libraries**: `scikit-learn`, `pandas`, `numpy`, `pickle`, `yaml`, `datasketch`

---

## Approach

1. **Importing Required Libraries and Packages**
2. **Open the config.yaml File**: This is a configuration file that can be edited according to your dataset.
3. **Read the JSON File**
4. **Clean the JSON File**:
   - Reset index in the data
   - Convert lists to integers
   - Remove rows above certain threshold values
   - Replace empty values with an empty list if any
   - Store the cleaned file
   - Calculate feature counts for scoring
5. **Model Training**:
   - Create a MinHashForest Model
   - Create an LSH graph object
   - Train the model
   - Save the model to a pickle file
6. **Seed Set Extension**:
   - Read the saved model
   - Read the seed set data
   - Retrieve the neighbors of the seed set from the LSH graph
   - Calculate the default click rate
   - Score the neighbors
   - Create and store the extension file
   - Find the increased click rate

---

## Modular Code Overview

1. **input**: Contains all the data for analysis, including a config file, the main JSON data file, the processed JSON file after cleaning, the `count_df.csv` file for feature counts, and the seed set CSV file.
2. **src**: The most important folder, containing modularized code for all the steps. It consists of `engine.py` and the `ml_pipeline` folder.
   - The `ml_pipeline` folder contains functions organized into different Python files, which are called inside `engine.py`.
3. **output**: Contains the best-fitted model and the extension CSV file. This model can be quickly loaded for future use, saving the need to retrain all models from the beginning.
4. **lib**: A reference folder that includes the original IPython notebook.

---

### Getting Started

Tested on Python 3.8.10

To install the dependencies run:
```
pip install -r requirements.txt
```


### Cleaning the dataset
To clean the data, run:
```
python processing.py
```
The cleaning step involves the following steps:
1) Convert all columns except column that can have multiple values to integer
2) Remove user records having number of values in the list columns more than a threshold value
3) Replace null values in list columns with empty list

### Train the model

To train the model, run:
```
python train.py
```

### Use the model to make predictions

To get extended users from a seed set, run:
```
python extend.py
```
