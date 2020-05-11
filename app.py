import sys
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import datetime

#Ekranda np ve pd'nin tüm satırı kısaltma olmadan gösterilmesi için yapılan ayarlar.
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

dataw = ""

print('Dataset Dosyası:')
print(os.listdir(str(os.path.abspath(os.getcwd())).replace('\\', '/') + "/data_en/"))
dataw += 'Dataset Dosyası:' + '\n' + str(os.listdir(str(os.path.abspath(os.getcwd())).replace('\\', '/') + "/data_en/"))

print("\nEmail verisi ilk 10 satır.")
df = pd.read_csv(str(os.path.abspath(os.getcwd())).replace('\\', '/') + "/data_en/" + 'emails.csv')
print(df.head(10))

print("\nSatır ve Sütün Sayısı")
print(df.shape)
dataw += '\n\n' + 'Satır ve Sütun Sayısı' + '\n' + str(df.shape)

def mailsayisi(x):
    mailsay = 0
    for index, row in df.iterrows():
        if row['spam'] == x:
            mailsay += 1
    return mailsay

print("\nMail ve Spam Mail Sayıları:")
print("Mail Sayısı:" + str(mailsayisi(0)))
print("Spam Sayısı:" + str(mailsayisi(1)))
dataw += '\n\nMail ve Spam Mail Sayıları:'
dataw += '\n' + 'Mail Sayısı:' + '\n' + str(mailsayisi(0))
dataw += '\n' + 'Spam Sayısı:' + '\n' + str(mailsayisi(1))

print("\nKolon Adları")
print(df.columns)
dataw += '\n\n' + 'Kolon Adları' + '\n' + str(df.columns)

print('')

print("Tekrarlı veriler siliniyor.")
df.drop_duplicates(inplace = True)
print('')

print("Tekrarlı veri silindikten sonraki satır/sütun sayısı")
print(df.shape)
dataw += '\n\n' + '(Tekrarsız) Satır ve Sutün Sayısı:' + '\n' + str(df.shape)

print("\nMail ve Spam Mail Sayıları:")
print("Mail Sayısı:" + str(mailsayisi(0)))
print("Spam Sayısı:" + str(mailsayisi(1)))
dataw += '\n\nMail ve Spam Mail Sayıları:'
dataw += '\n' + 'Mail Sayısı:' + '\n' + str(mailsayisi(0))
dataw += '\n' + 'Spam Sayısı:' + '\n' + str(mailsayisi(1))

print("\nHer bir kolon için eksik/None veri sayısı")
print(df.isnull().sum())
dataw += '\n\n' + 'Her bir kolon için eksik/None veri sayısı:' + '\n' + str(df.isnull().sum())

nltk.download('stopwords')


# Tokenizasyon (tokenler listesi kullanılacak), analiz için.
# 1.Noktalama işaretleri [!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]
# 2.Gereksiz sözcükler stopword'ler kaldırılacak. (data).
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
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    # 3
    return clean_words

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

print("Tokenizasyon yapılıyor.")
tokenizasyon = df['text'].head().apply(process_text)
print(tokenizasyon)
dataw += '\n\n' + 'Tokenizasyon:' + '\n' + str(tokenizasyon)

print("Metni bir token sayımı matrisine dönüştürüyoruz.")
from sklearn.feature_extraction.text import CountVectorizer
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])
print("Dönüştürme tamamlandı.")
dataw += '\n\n' + 'Metin bir token sayım matrisine dönüştürüldü.'

print("data'nın 80%'i training ve 20%'si test data setleri olacak")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['spam'], test_size = 0.20, random_state = 0)
print("Training ve Test setleri tamamlandı.")
dataw += '\n\n' + 'Traning %80 / Test %20 setleri tamamlandı.'

#messages_bow 'un shape'sini yazdır.
print('Message_Bow Shape:')
print(messages_bow.shape)
dataw += '\n\n' + 'Message_Bow Shape:\n' + str(messages_bow.shape)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

print("\nYazdırılan Tahmini Değerler:")
print(classifier.predict(X_train))
dataw += '\n\n' + 'Yazdırılan Tahmini Değerler:' + '\n' + str(classifier.predict(X_train))

print("\nYazdırılan Gerçek Değerler:")
print(y_train.values)
dataw += '\n\n' + 'Yazdırılan Gerçek Değerler:' + '\n' + str(y_train.values)

#Model'in training veri setine (data set) göre değerlendirilmesi.
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_train)

print("\nSınıflandırma Raporu:")
print(classification_report(y_train,pred))
dataw += '\n\n\n' + 'Sınıflandırma Raporu:' + '\n' + str(classification_report(y_train,pred))

print('Karışıklık Matrisi: \n', confusion_matrix(y_train,pred))
dataw += '\n\n' + 'Karışıklık Matrisi:' + str(confusion_matrix(y_train,pred))

print()
print('Accuracy: ', accuracy_score(y_train,pred))
print()
dataw += '\n\n' + 'Accuracy: ' + str(accuracy_score(y_train,pred))

print('\nModelin Test Datasete göre değerlendirilmesi:')
dataw += '\n\n' + 'Modelin Test Datasete göre değerlendirilmesi:'

#Tahmini değerler
print('\nTahmini değer: \n',classifier.predict(X_test))
dataw += '\n\n' + 'Tahmini değer: \n' + str(classifier.predict(X_test))

#Gerçek değerler
print('\nGerçek değer: \n',y_test.values)
dataw += '\n\n' + 'Gerçek değer: \n' + str(y_test.values)

#Model'in test data set e göre değerlendirilmesi

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
pred = classifier.predict(X_test)

print()
print(classification_report(y_test,pred))
dataw += '\n\n' + str(classification_report(y_test,pred))

print('Karışıklık Matrisi: \n', confusion_matrix(y_test,pred))
dataw += '\n\n' + 'Karışıklık Matrisi: \n' + str(confusion_matrix(y_test,pred))

print()

print('Accuracy: ', accuracy_score(y_test,pred))
dataw += '\n\n' + 'Accuracy: ' + str(accuracy_score(y_test,pred))

currentDT = datetime.datetime.now()
dosya_sonuc = "sonuc_" + str(currentDT.strftime("%d-%m-%Y %H-%M-%S")) + ".txt"
str1 = str(os.path.abspath(os.getcwd())).replace('\\', '/') + "/log_en/" + dosya_sonuc

file = open(str1, "w")
n = file.write(dataw)
file.close()

plotPerColumnDistribution(df, 10, 5)