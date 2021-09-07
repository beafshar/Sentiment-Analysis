import pandas as pd
import csv



def txt_to_csv(txt_file, label, output):

    file = open(txt_file, 'r')
    lines = file.readlines()


    labels = [label]* len(lines)


    dict = {'review': lines, 'sentiment': labels}  

    df = pd.DataFrame(dict) 


    df.to_csv(output + '.csv') 


txt_to_csv("../socialnetworkanalysiic_socialnetwork_post.csv", -1, "test.csv")
# txt_to_csv("neutralNew.txt", 0 , "neutral")
# txt_to_csv("positiveEyesNew.txt", 1, "positiveEyes")
# txt_to_csv("positiveFaceNew.txt", 1, "positiveFace")