# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:58:28 2024

@author: CJ
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from bs4 import BeautifulSoup
import requests
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertModel, AdamW
import torch.nn as nn
import torch.optim as optim
import time
import os

# Check if GPU is available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using device: {device}')

# -------------------------
# 1. Data Loading and Preprocessing
# -------------------------

# Load CSV files
profiles_df = pd.read_csv(r'C:\Users\CJ\Documents\Python Projects\player_descriptions\nfl_draft_profiles.csv')
prospects_df = pd.read_csv(r'C:\Users\CJ\Documents\Python Projects\player_descriptions\nfl_draft_prospects.csv')
id_df = pd.read_csv(r'C:\Users\CJ\Documents\Python Projects\player_descriptions\ids.csv')

# Merge profiles and prospects on 'player_id'
df_merged = profiles_df.merge(prospects_df, on='player_id', how='inner')

# Fill NaN 'round' with 8 (undrafted)
df_merged['round'] = df_merged['round'].fillna(8)

# Select relevant columns
df = df_merged.loc[:, ['player_id', 'player_name_x', 'text1', 'text2', 'text3', 'text4', 'overall']]

# Drop rows with less than 4 non-NA values and fill remaining NAs with empty strings
df = df.dropna(thresh=4)
df.fillna('', inplace=True)
df.reset_index(drop=True, inplace=True)

# Consolidate multiple text fields into a single 'description'
def consolidate_description(row):
    for text_field in ['text1', 'text2', 'text3', 'text4']:
        if row[text_field]:
            return row[text_field]
    return ''

df['description'] = df.apply(consolidate_description, axis=1)

# Abstract player names and clean descriptions
def clean_description(row):
    name_parts = row['player_name_x'].split(' ')
    if len(name_parts) >= 2:
        first, last = name_parts[0], name_parts[1]
    else:
        first, last = name_parts[0], ''
    description = row['description']
    description = re.sub(re.escape(first), 'FIRST NAME', description)
    if last:
        description = re.sub(re.escape(last), 'LAST NAME', description)
    # Remove HTML tags and newline characters
    description = re.sub(r'\n|\r', '', description)
    description = re.sub(r'<.*?>', '', description)
    return description

df['description'] = df.apply(clean_description, axis=1)

# Replace empty 'overall' with 300 (undrafted)
df['overall'].replace('', 300, inplace=True)
df['overall'] = df['overall'].astype(float)  # Ensure 'overall' is float for regression

# -------------------------
# 2. Web Scraping Draft Pick Values
# -------------------------

def scrape_draft_values(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    
    # Adjust the selector based on actual webpage structure
    pick_values = soup.select("tr + tr td")
    pick_num = [title.text.strip() for title in pick_values]
    
    pick_pos = [300]  # Starting with overall value 300 for undrafted
    pick_val = [0.0]
    
    for i in range(len(pick_num)):
        try:
            num = float(pick_num[i].replace(',', ''))
            if i % 2 == 0:
                pick_pos.append(num)
            else:
                pick_val.append(num)
        except ValueError:
            continue  # Skip non-numeric entries
    
    return pd.DataFrame({'overall': pick_pos, 'draft_value': pick_val})

# URL for draft pick values
url = "https://www.nbcdfw.com/news/sports/nfl/nfl-draft-pick-trade-value-chart-breakdown/3510384/"
pick_values_df = scrape_draft_values(url)

# Merge draft pick values into main dataframe
df2 = df.merge(pick_values_df, on='overall', how='left')
df2.fillna(0.0, inplace=True)

# -------------------------
# 3. Preparing Data for Modeling
# -------------------------

# Features and target
X = df2['description']
y = df2['draft_value']

# Train-test split
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=42)

# -------------------------
# 4. BERT-Based Modeling with DistilBERT
# -------------------------

# Parameters
BATCH_SIZE = 16  # Reduced batch size for better performance on limited hardware
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128  # Maximum token length

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenization function
def tokenize_texts(texts, tokenizer, max_length=MAX_LENGTH):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# Tokenize training and testing data
print("Tokenizing training data...")
encoded_train = tokenize_texts(train_x, tokenizer)
print("Tokenizing testing data...")
encoded_test = tokenize_texts(test_x, tokenizer)

# Convert labels to tensors
labels_train = torch.tensor(train_y.tolist(), dtype=torch.float)
labels_test = torch.tensor(test_y.tolist(), dtype=torch.float)

# Create TensorDatasets
train_dataset = TensorDataset(
    encoded_train['input_ids'],
    encoded_train['attention_mask'],
    labels_train
)

test_dataset = TensorDataset(
    encoded_test['input_ids'],
    encoded_test['attention_mask'],
    labels_test
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load pre-trained DistilBERT model
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

# Define the regression head
class BertRegressor(nn.Module):
    def __init__(self, bert_model, hidden_size=768):
        super(BertRegressor, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_size)
        dropout_output = self.dropout(cls_output)
        linear_output = self.linear(dropout_output)  # Shape: (batch_size, 1)
        return linear_output.squeeze(-1)  # Shape: (batch_size)

# Instantiate the model
regressor = BertRegressor(model)
regressor.to(device)

# Define optimizer and loss function
optimizer = AdamW(regressor.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

# -------------------------
# 5. Training the Model
# -------------------------

print("Starting training...")
start_time = time.time()

for epoch in range(EPOCHS):
    regressor.train()
    epoch_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = regressor(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}')

end_time = time.time()
print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")

# -------------------------
# 6. Evaluating the Model
# -------------------------

print("Evaluating the model on test data...")
regressor.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = regressor(input_ids, attention_mask)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(labels.cpu().numpy())

# Convert lists to numpy arrays
predictions = np.array(predictions)
actuals = np.array(actuals)

# Calculate evaluation metrics
mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actuals, predictions)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'R-squared (R2 ): {r2:.4f}')

# -------------------------
# 7. Mapping Predictions to Draft Positions
# -------------------------

# Function to map draft_value to closest overall pick
def map_to_draft_position(pred_value, pick_values_df):
    # Calculate absolute differences
    diffs = abs(pick_values_df['draft_value'] - pred_value)
    # Find the index of the minimum difference
    min_idx = diffs.idxmin()
    return pick_values_df.loc[min_idx, 'overall']

# Apply the mapping to predictions and actuals
pred_pos = [map_to_draft_position(pred, pick_values_df) for pred in predictions]
true_pos = [map_to_draft_position(true, pick_values_df) for true in actuals]

# Create a results DataFrame
draft_pos_results = pd.DataFrame({
    'True Draft Value': actuals,
    'Predicted Draft Value': predictions,
    'True Draft Position': true_pos,
    'Predicted Draft Position': pred_pos
})

print(draft_pos_results.head())

# Save the results to a CSV file
output_path = r'C:\Users\CJ\Documents\Python Projects\player_descriptions\draft_pos_results_BERT.csv'
draft_pos_results.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")

# Additional Evaluation

import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot of True vs Predicted Draft Values
plt.figure(figsize=(8,6))
sns.scatterplot(x=true_pos, y=pred_pos)
plt.xlabel('True Draft Position')
plt.ylabel('Predicted Draft Position')
plt.title('True vs Predicted Draft Positions')
plt.plot([min(true_pos), max(true_pos)], [min(true_pos), max(true_pos)], color='red')  # Diagonal line
plt.show()

# Histogram of prediction errors
errors = predictions - actuals
plt.figure(figsize=(8,6))
sns.histplot(errors, bins=30, kde=True)
plt.xlabel('Prediction Error (Predicted - True)')
plt.title('Distribution of Prediction Errors')
plt.show()
