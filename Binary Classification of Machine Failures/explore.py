import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.impute import SimpleImputer

train_file_path = r"C:\Users\user\Python Stuff\Kaggle Projects\Binary Classification of Machine Failures\input\train.csv"
test_file_path = r"C:\Users\user\Python Stuff\Kaggle Projects\Binary Classification of Machine Failures\input\test.csv"

train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)

#impute + ordinal/one_hot

