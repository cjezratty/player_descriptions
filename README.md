# player_descriptions
Predicting draft position of NFL players from their draft description only
# NFL Draft Pick Prediction Using Text Data

This project explores the use of **Large Language Model (LLM) techniques** such as tokenizing, encoding, word embeddings, and transfer learning with BERT to predict the NFL draft position of a player based solely on text descriptions written by scouts. The primary goal was to gain hands-on experience with LLMs while exploring their practical applications.

---

## Table of Contents
1. [Overview](#overview)
2. [Files and Data](#files-and-data)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Takeaways](#takeaways)
6. [Installation](#installation)
7. [Usage](#usage)
8. [License](#license)

---

## Overview

The project attempts to predict the **overall draft position** of NFL players using text descriptions from scouting blurbs, ignoring all other predictive variables. Two approaches were implemented:
1. **Word Embeddings**: Using preprocessed text and embedding layers.
2. **BERT**: Using a pretrained BERT model with transfer learning.

### Goals
- Learn and apply foundational techniques in Natural Language Processing (NLP), including word tokenization, encoding, and word embeddings.
- Experiment with BERT for transfer learning on text data.
- Evaluate and compare the predictive performance of these models.

---

## Files and Data

### **Data Files**
- `nfl_draft_profiles.csv`: Contains text descriptions for each player.
- `nfl_draft_prospects.csv`: Metadata about NFL draft prospects.
- `ids.csv`: Mapping of player IDs to dataset entries.

### **Scripts**
1. `player_descriptions_WE.py`:
   - Implements word preprocessing, tokenization, and word embeddings.
   - Trains both a regression model and a random forest model to predict the overall draft pick.
2. `draft_round_predictor_BERT.py`:
   - Explores BERT's pretrained model and transfer learning for the same prediction task.

### **Results Files**
- `draft_pos_results.csv`: Results from the word embedding-based models.
- `draft_pos_results_BERT.csv`: Results from the BERT-based model.

---

## Methodology

1. **Data Preprocessing**:
   - Tokenization, encoding, and text cleaning techniques were applied to prepare scouting blurbs.
   - For BERT, minimal preprocessing was done to leverage the pretrained tokenizer.

2. **Models**:
   - **Word Embeddings**:
     - Used embeddings to transform text data into numerical vectors.
     - Trained a regression model and a random forest model.
   - **BERT**:
     - Leveraged BERT’s pretrained embeddings and fine-tuned it for predicting draft positions.

3. **Evaluation**:
   - Predicted the overall draft pick and recorded results in CSV files for analysis.

---

## Results

### Word Embedding Models
- Produced **reasonable predictions**, but with a bias toward predicting middle-round picks.
- Demonstrated the potential of simple word embeddings for regression tasks.

### BERT Model
- Performed **poorly** due to the complexity of the model and insufficient preprocessing.
- Highlighted challenges in applying large models to smaller datasets without robust preprocessing.

---

## Takeaways

- **Key Learnings**:
  - Hands-on experience with tokenization, encoding, word embeddings, and transfer learning using BERT.
  - Explored the challenges of working with large models like BERT and learned the importance of preprocessing.

- **Future Improvements**:
  - Enhance text preprocessing for BERT to better utilize the model’s capabilities.
  - Experiment with smaller pretrained models to avoid overfitting.

