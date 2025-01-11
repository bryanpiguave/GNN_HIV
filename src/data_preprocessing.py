"""
This script preprocesses an HIV dataset by performing the following steps:
1. Changes the working directory to the parent directory.
2. Defines a function to split the data into training and testing sets.
3. Reads the raw HIV data from a CSV file.
4. Splits the data into training and testing sets.
5. Saves the training and testing sets to CSV files.
6. Reads the training data back from the CSV file.
7. Counts the number of positive and negative samples in the training data.
8. Applies oversampling to balance the dataset by replicating the positive class.
9. Appends the replicated data to the original training data.
10. Shuffles the dataset and resets the index.
11. Saves the oversampled training data to a CSV file.

Functions:
- split_data(data, test_size=0.2): Splits the input data into training and testing sets.

Main function:
- main(): Preprocesses the HIV data and saves the processed data to CSV files.

Usage:
Run this script directly to preprocess the HIV data.

Example:
$ python preprocess_hiv_data.py
"""


import pandas as pd 
from sklearn.model_selection import train_test_split
import os
from rdkit import Chem
import tqdm 
os.chdir("..")

def split_data(data, test_size=0.2):
    train, test = train_test_split(data, test_size=test_size)
    return train, test

def main():
    print('*'*50)
    print("Preprocessing data")
    data = pd.read_csv('data/raw/HIV.csv')

    data = data.dropna()
    #Clean invalid smiles 
    for index, row in tqdm.tqdm(data.iterrows(), total=data.shape[0]):        
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            print(f"Invalid SMILES: {row['smiles']}")
            data.drop(index, inplace=True)
    train, test = split_data(data)
    train.to_csv('data/raw/HIV_train.csv', index=False)
    test.to_csv('data/raw/HIV_test.csv', index=False)
    data = pd.read_csv("data/raw/HIV_train.csv")
    data["HIV_active"].value_counts()
    # Applying oversampling
    # Check how many additional samples we need
    neg_class = data["HIV_active"].value_counts()[0]
    pos_class = data["HIV_active"].value_counts()[1]
    multiplier = int(neg_class/pos_class) - 1

    # Replicate the dataset for the positive class
    replicated_pos = [data[data["HIV_active"] == 1]]*multiplier
    # Append replicated data
    data = data._append(replicated_pos,
                        ignore_index=True)
    print(data.shape)
    # Shuffle dataset
    data = data.sample(frac=1).reset_index(drop=True)
    # Re-assign index (This is our ID later)
    data.head()
    # Save
    data.to_csv("data/raw/HIV_train_oversampled.csv", index=False)
    return 0 

if __name__ == "__main__":
    main()