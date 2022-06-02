import pandas as pd
import numpy as np
import ast
import itertools

# dummy data
df = pd.DataFrame({
    "ID": [1,2,3],
    "production_companies": ["[{'1': 'Paramount Pictures', '2': 4}, {'1': 'United Artists', '2': 60}, {'1': 'Metro-Goldwyn-Mayer (MGM)', '2': 8411}]", np.nan, "[{'1': 'Walt Disney Pictures', '2': 2}]"]
})

# remove the nans
df.dropna(inplace=True)

print(df["production_companies"] )

# convert the strings into lists
df["production_companies"] = df["production_companies"].apply(lambda x: ast.literal_eval(x))

def


