from datasets import load_dataset
import pandas as pd 

bots = pd.read_csv('bot_detection_data.csv')
npl = load_dataset("junaid1993/Dataset_Bot_Detection")

twitter = load_dataset("airt-ml/twitter-human-bots")
twitter = twitter['train'].to_pandas() # convert to pandas dataframe
npl = npl['train'].to_pandas() # convert to pandas dataframe


print(f'Bot data columns:\n {bots.columns}\n\n')
print(f'npl data columns:\n {npl.columns}\n\n')
print(f'twitter data columns:\n {twitter.columns}')
# ----- Standardise column names & types ----------
#Align names.



# -------Handle Missing ID's --------



# ------Merge dataframes onto column. ----------



