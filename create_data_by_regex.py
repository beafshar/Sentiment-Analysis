import pandas as pd
import csv
import re


def txt_to_csv(txt_file, label, output):

    with open(txt_file, 'r') as file:
        data = file.read()
    lines = re.split("\*\*\*\*\*\*\*\*\*\*", data)
    labels = [label]* len(lines)



    dict = {'review': lines, 'sentiment': labels}  

    df = pd.DataFrame(dict) 


    df.to_csv('data csv/' + output + '.csv') 


# txt_to_csv('sentiment_dateset_v_1/negative_v_1.txt', -1, "negative")
# txt_to_csv("sentiment_dateset_v_1/neutral_v_1.txt", 0 , "neutral")
# txt_to_csv("sentiment_dateset_v_1/positive_v_1.txt", 1, "positive")


df2 = pd.read_csv("data csv/negative.csv")

print(df2["review"].iloc[0])
df1 = pd.read_csv("data csv/positive.csv")

print(df1["review"].iloc[0])
