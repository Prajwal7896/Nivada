import pandas as pd 

data = pd.read_csv("final_complaints_dataset_with_categories.csv")

print(data["category"].value_counts())