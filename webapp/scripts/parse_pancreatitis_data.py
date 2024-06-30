# Helper script to convert the pancreatitis excel file to a features and outcomes csv file
from pathlib import Path

import pandas as pd

input_file = "/Users/thomasvetterli/Desktop/Quantimage_Project/Patient_Data/pancreatitis/Pancreatitis_all_data.xlsx" 
output_dir = Path("/Users/thomasvetterli/Desktop/Quantimage_Project/Patient_Data/pancreatitis")

df = pd.read_excel(input_file)

# removing columns that are labels + unknown features
df_of_interest = df.iloc[:, :120]
columns = df_of_interest.iloc[2]
df_of_interest = df_of_interest.iloc[3:]
df_of_interest.columns = columns 
df_of_interest.rename(columns={"Patient_ID": "PatientID"}, inplace=True)

# Removing date columns
date_columns = [i for i in df_of_interest.columns if "date" in i.lower()]
df_of_interest = df_of_interest.drop(date_columns, axis=1)

# putting M and F to lowercase
df_of_interest["Gender (M/F)"] = df_of_interest["Gender (M/F)"].str.lower()

# creating an outcome column from the ransom score
ADMISSION_RANSON = "Ranson score admission"
RANSON_48H = "Ranson score 48h"
RANSON_TOTAL = "Total ranson score "


outcome_df = df_of_interest[["PatientID", ADMISSION_RANSON]]

# Binarizing the ranson score
outcome_df[ADMISSION_RANSON] = (outcome_df[ADMISSION_RANSON] >= 2).astype(int)
outcome_df.rename(columns={ADMISSION_RANSON: "Outcome"}, inplace=True)

# Removing the labels from the feature df
df_of_interest.drop([ADMISSION_RANSON, RANSON_48H, RANSON_TOTAL], axis=1, inplace=True)

df_of_interest.to_csv(output_dir / "pancreatitis_features.csv", index=False)
outcome_df.to_csv(output_dir / "pancreatitis_outcomes_ransom_admission_larger_2.csv", index=False)