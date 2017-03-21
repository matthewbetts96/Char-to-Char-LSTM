import os
import sys
import imp
import sqlite3
import tflearn
import tensorflow as tf
from tflearn.data_utils import *
 
path = "sentences.txt"
path = path.encode('utf-8')
 
sql = sqlite3.connect('reddit_comments.db')
cur = sql.cursor()
 
os.remove('sentences.txt')
 
print(sys.version[0])
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
 
alphabet = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","0","1","2","3","4","5","6","7","8","9"," ",",",".","!",'"',"@","<",">","[","]","{","}","(",")",":",";","%","£","$","^","&","*","'","/","|","~","#","+","-","=","_"]
with open('sentences.txt', 'w') as file:
    for row in cur.execute("SELECT * FROM posts WHERE (subreddit = 'The_Donald')"):
        body, author, subreddit, subredditID = (row)   
        chardWords = list(body)
        for index, item in enumerate(chardWords):
            if item not in alphabet:
                chardWords[index] = " "
        joinedWords = "".join(chardWords)
        file.write(author + ":")
        file.write("\n")
        file.write(joinedWords)
        file.write("\n")
 
#max length
maxlen = 25
 
#vectorising
X, Y, char_idx = textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step = 10)
 
g = tflearn.input_data(shape = [None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq = True)
g = tflearn.dropout(g, 0.2)
g = tflearn.lstm(g, 512, return_seq = True)
g = tflearn.dropout(g, 0.2)
g = tflearn.lstm(g, 512)
g = tflearn.fully_connected(g, len(char_idx), activation = 'softmax')
g = tflearn.regression(g, optimizer = 'adam', loss= 'categorical_crossentropy', learning_rate = 0.001)
 
m = tflearn.SequenceGenerator(g, dictionary = char_idx, seq_maxlen=maxlen, clip_gradients = 5.0, tensorboard_verbose=1)
#seed = random_sequence_from_textfile(path, maxlen)
seed = "trump is"
 
m.fit(X, Y, validation_set = 0.05, batch_size = 50, n_epoch = 30, show_metric = True, run_id = 'sentences')
 
print("-------------")
print('Seed = ' + seed)
print("Temp = 1")
print(m.generate(1000, temperature = 1.0, seq_seed = seed))
print("-------------")
print("Temp = 0.9")
print(m.generate(1000, temperature = 0.9, seq_seed = seed))
print("-------------")
print("Temp = 0.8")
print(m.generate(1000, temperature = 0.8, seq_seed = seed))
print("-------------")
print("Temp = 0.7")
print(m.generate(1000, temperature = 0.7, seq_seed = seed))
print("-------------")
print("Temp = 0.6")
print(m.generate(1000, temperature = 0.6, seq_seed = seed))
print("-------------")
print("Temp = 0.5")
print(m.generate(1000, temperature = 0.5, seq_seed = seed))
print("-------------")
print("Temp = 0.25")
print(m.generate(1000, temperature = 0.25, seq_seed = seed))
print("-------------")