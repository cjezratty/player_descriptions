# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:50:21 2024

@author: CJ
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

# Read the CSV files
profiles_df = pd.read_csv(r'C:\Users\CJ\Documents\Python Projects\player_descriptions\nfl_draft_profiles.csv')
prospects_df = pd.read_csv(r'C:\Users\CJ\Documents\Python Projects\player_descriptions\nfl_draft_prospects.csv')
id_df = pd.read_csv(r'C:\Users\CJ\Documents\Python Projects\player_descriptions\ids.csv')

# Join profiles and prospects on the player id attribute
df_merged = profiles_df.merge(prospects_df, on='player_id', how='inner')

# Changing NaN to undrafted by signifying round 8
df_merged['round'] = df_merged['round'].fillna(8)

# Clean out the data that has no descriptions
df = df_merged.loc[:, ['player_id', 'player_name_x', 'text1', 'text2', 'text3', 'text4', 'overall']]
df = df.dropna(thresh=4)
df.fillna('', inplace=True)
df.index = range(0, df.shape[0])

# Consolidate to just one description per player
description = []
for i in range(0, df.shape[0]):
    if df.loc[i, 'text1'] != '':
        description.append(df.loc[i, 'text1'])
    elif df.loc[i, 'text2'] != '':
        description.append(df.loc[i, 'text2'])
    elif df.loc[i, 'text3'] != '':
        description.append(df.loc[i, 'text3'])
    else:
        description.append(df.loc[i, 'text4'])

df['description'] = description

# Abstract the player names from the descriptions
for i in range(0, df.shape[0]):
    name_parts = df.loc[i, 'player_name_x'].split(' ')
    if len(name_parts) >= 2:
        first, last = name_parts[0], name_parts[1]
    else:
        first, last = name_parts[0], ''
    df.loc[i, 'description'] = re.sub(first, 'FIRST NAME', df.loc[i, 'description'])
    if last:
        df.loc[i, 'description'] = re.sub(last, 'LAST NAME', df.loc[i, 'description'])
    df.loc[i, 'description'] = re.sub(r'\n', '', df.loc[i, 'description'])
    df.loc[i, 'description'] = re.sub(r'\r', '', df.loc[i, 'description'])
    df.loc[i, 'description'] = re.sub(r'<p>', '', df.loc[i, 'description'])
    df.loc[i, 'description'] = re.sub(r'<b>', '', df.loc[i, 'description'])
    df.loc[i, 'description'] = re.sub(r'</b>', '', df.loc[i, 'description'])
    df.loc[i, 'description'] = re.sub(r'</p>', '', df.loc[i, 'description'])
    df.loc[i, 'description'] = re.sub(r'<strong>', '', df.loc[i, 'description'])
    df.loc[i, 'description'] = re.sub(r':</strong>', '', df.loc[i, 'description'])

# Let overall 300 indicate an undrafted player
df['overall'].replace('', 300, inplace=True)

# ---------------------------------------------------------------
# Web scraping to get draft pick values
from bs4 import BeautifulSoup
import requests

url = "https://www.nbcdfw.com/news/sports/nfl/nfl-draft-pick-trade-value-chart-breakdown/3510384/"
res = requests.get(url)
soup = BeautifulSoup(res.text, features="html.parser")

pick_values = soup.select("tr + tr td")
pick_num = [title.text for title in pick_values]

pick_pos = [300]  # Starting with overall value 300 for undrafted
pick_val = [0]
for i in range(len(pick_num)):
    try:
        num = float(pick_num[i].replace(',', ''))
    except ValueError:
        continue
    if i % 2 == 0:
        pick_pos.append(num)
    else:
        pick_val.append(num)

pick_values_df = pd.DataFrame({'overall': pick_pos, 'draft_value': pick_val})

# Merge draft pick value into our main dataframe using the 'overall' key
df2 = df.merge(pick_values_df, on='overall', how='left')
df2.fillna(0.0, inplace=True)

# ---------------------------------------------------------------
# Prepare data for modeling
X = df2['description']
y = df2['draft_value']

# ---------------------------------------------------------------
# GloVe Embeddings preparation
from gensim.models import KeyedVectors
import os

def add_header_to_glove(input_path, output_path):
    # Count the number of lines in the input file to determine the vocabulary size
    vocab_size = sum(1 for line in open(input_path, encoding='utf-8'))
    # Get the dimensionality of the word vectors from the first line
    with open(input_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        vector_dimensionality = len(first_line.split()) - 1  # Subtract 1 for the word
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write(f"{vocab_size} {vector_dimensionality}\n")
            # Write the first line and the rest
            out_f.write(first_line + '\n')
            for line in f:
                out_f.write(line)

# Example usage of the header function:
input_path = r'C:\Users\CJ\Documents\Python Projects\player_descriptions\glove.6B.300d.txt'
output_path = r'C:\Users\CJ\Documents\Python Projects\player_descriptions\glove.6B.300d_header.txt'
# Uncomment if header addition is needed:
# add_header_to_glove(input_path, output_path)

# Load pre-trained GloVe word vectors (adjust path and dimensionality as needed)
glove_path = r'C:\Users\CJ\Documents\Python Projects\player_descriptions\glove.6B.100d_header.txt'
word_vectors = KeyedVectors.load_word2vec_format(glove_path, binary=False)

# Tokenize descriptions and convert to word embeddings (averaging the word vectors)
def description_to_embedding(description):
    words = description.split()
    embedding = np.zeros(100)  # Assuming 100-dimensional GloVe embeddings
    count = 0
    for word in words:
        if word in word_vectors.key_to_index:
            embedding += word_vectors.get_vector(word)
            count += 1
    if count > 0:
        return embedding / count
    else:
        return embedding

embeddings = X.apply(description_to_embedding)

# ---------------------------------------------------------------
# Split the data into training and test sets
X_embeddings = np.vstack(embeddings)
y_values = df2['draft_value']
train_x, test_x, train_y, test_y = train_test_split(X_embeddings, y_values, test_size=0.25, random_state=42)

# ---------------------------------------------------------------
# Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(train_x, train_y)
y_pred_lin = lin_model.predict(test_x)
lin_mse = mean_squared_error(test_y, y_pred_lin)
print(f'Linear Regression Mean Squared Error: {lin_mse}')

# ---------------------------------------------------------------
# Random Forest Model
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_regressor.fit(train_x, np.array(train_y))
y_pred_rf_train = rf_regressor.predict(train_x)
y_pred_rf_test = rf_regressor.predict(test_x)

train_rmse = mean_squared_error(train_y, y_pred_rf_train, squared=False)
test_rmse = mean_squared_error(test_y, y_pred_rf_test, squared=False)

print("Random Forest Train RMSE:", train_rmse)
print("Random Forest Test RMSE:", test_rmse)

# ---------------------------------------------------------------
# Map model predictions (and true values) back to draft positions using the pick_values_df.
# Here, we use the Random Forest predictions, but you could switch to linear if preferred.
pred_pos = []
true_pos = []

# To ensure sequential integer indexing, reset the index for test_y and predictions.
test_y_reset = test_y.reset_index(drop=True)
y_pred_rf_test = pd.Series(y_pred_rf_test).reset_index(drop=True)

for i in range(len(test_y_reset)):
    # Use .iloc to avoid KeyError - access by integer position
    true_val = test_y_reset.iloc[i]
    spot = pick_values_df.loc[pick_values_df['draft_value'] == true_val]
    if not spot.empty:
        true_pos_value = spot.iloc[0, 0]  # overall (pick position)
    else:
        true_pos_value = np.nan  # In case there is no exact match
    true_pos.append(true_pos_value)
    
    # Calculate differences between predicted value and available draft values
    pick_values_df['diffs'] = abs(y_pred_rf_test.iloc[i] - pick_values_df['draft_value'])
    # Find the index of the closest draft_value
    temp_pos = pick_values_df['diffs'].idxmin()
    # Append the corresponding overall pick number as the predicted position
    pred_pos.append(pick_values_df.loc[temp_pos, 'overall'])

draft_pos_results = pd.DataFrame({'True Position': true_pos, 'Predicted Position': pred_pos})
print(draft_pos_results.head())

# save the results to a CSV file
draft_pos_results.to_csv(r'C:\Users\CJ\Documents\Python Projects\player_descriptions\draft_pos_results.csv', index=False)
