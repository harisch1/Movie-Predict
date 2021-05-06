import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preProcess(name): 
    file = open(name, 'r', encoding='utf8')
    df=pd.read_csv(name)
    pd.set_option('mode.chained_assignment', None)
    #rename and drop useless columns
    df = df.rename(columns={"type": "length"})
    df = df.drop(columns =['teleplay_id'])

   #length in int
    df['length'].replace('long', 3, inplace=True)
    df['length'].replace('medium', 2, inplace=True)
    df['length'].replace('short', 1, inplace=True)

    #Move genre in place

    for i in range (len(df)):
        gen = str(df['genre'][i]).split(',')
        if (df['length'][i] not in [0,1,2]):
            try:
                gen.append(df['length'][i])
                df['length'][i] = 0
            #print(gen)
            except (SettingWithCopyWarning):
                print('Exception in processing')
        df['genre'][i] = gen

    #List Genre
    genre = []
    for i in range(len(df)):
        for g in df['genre'][i]:
            if (g not in genre and g != '' and g != 'nan'):
                genre.append(g)

    #Seprate Genre
    df[genre] = 0
    for i in range(len(df)):
        for g in df['genre'][i]:
            if g!='nan':
                df[g][i] = 1

    #drop genre, empty ratings and blanks
    df = df.drop(columns = ['genre'])
    df['episodes'].replace('Unknown', 0, inplace = True)
    if (name == "Teleplay.csv"):
        df['rating'].replace('', np.nan, inplace=True)
        df.dropna(subset=['rating'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    #Name Length
    names = []
    l = len(df)
    print()
    
    df = df.drop(columns =['name'])
    df = df.apply(pd.to_numeric)
    # scale
    #df = df.apply(lambda x: x / x.max())
    df = df.fillna(0)
    #df = df.dropna(inplace = True)

    
    X = df.drop(columns =['rating'])
    if (name == "Teleplay.csv"):
        Y = df['rating']
        return np.array(X, dtype=float), np.array(Y, dtype=float)
    return np.array(X, dtype=float)