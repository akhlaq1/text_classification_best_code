import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import datetime
import _pickle as cPickle
import joblib

#Ekranda np ve pd'nin tüm satırı kısaltma olmadan gösterilmesi için yapılan ayarlar.
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def process_text(text):
    '''
    Yapılacaklar:
    1. Noktalama işaretleri silinecek.
    2. Stopword'ler silinecek.
    3. Temizlenmiş kelimeler clean_words döndürülecek.
    '''

    # 1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    # 2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('turkish')]

    # 3
    return clean_words

class Predictor():
    def __init__(self,inputFile, outputFile):
        self.inputFile = inputFile
        self.outputFile = outputFile
        self.df = pd.read_csv(inputFile)
        
    def predictor_func(self):
        dataw = ""

        # print('Dataset Dosyası:')
        # print(os.listdir(str(os.path.abspath(os.getcwd())).replace('\\', '/') + "/data_tr/"))
        # dataw += 'Dataset Dosyası:' + '\n' + str(os.listdir(str(os.path.abspath(os.getcwd())).replace('\\', '/') + "/data_tr/"))

        print("\nEmail verisi ilk 10 satır.")
        
        df = self.df
        
        print(df.head(10))

        print("\nSatır ve Sütün Sayısı")
        print(df.shape)
        dataw += '\n\n' + 'Satır ve Sütun Sayısı' + '\n' + str(df.shape)


        print(df.columns)
        dataw += '\n\n' + 'Kolon Adları' + '\n' + str(df.columns)

        df.drop_duplicates(inplace = True)
        print('')


        nltk.download('stopwords')


        with open('classifier_vector.pkl', 'rb') as fid:
            vectors,classifier = cPickle.load(fid)
            
        messages_bow = vectors.transform(df['text']) 

        output = classifier.predict(messages_bow)

        df['predict'] = output 
        
        df.to_csv(self.outputFile)


    # def mailsayisi(self,x):
    #     mailsay = 0
    #     for index, row in self.df.iterrows():
    #         if row['spam'] == x:
    #             mailsay += 1
    #     return mailsay


