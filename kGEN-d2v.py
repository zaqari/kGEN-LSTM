import pandas as pd
import numpy as np
import gensim
from nltk.stem import WordNetLemmatizer

import keras
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Flatten
#from keras.layers.wrappers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Activation

########################################
########LEMMATIZATION SET-UP
########################################
lem=WordNetLemmatizer()


########################################
########WORD VEC VARIABLES
########################################
d2v_model_path='./1-data/_vec-models/d2v-200k-1.bin'
a2v_model_path='./1-data/_vec-models/a2v-200k-1.bin'

d2vs=gensim.models.Word2Vec.load(d2v_model_path)
a2vs=gensim.models.Word2Vec.load(a2v_model_path)


########################################
########IMPORT DOCS
########################################
train_data = './1-data/train_data14v5-mob.csv'
test_data = './1-data/test_data14v5-mob.csv'
df_train = pd.read_csv(train_data, skipinitialspace=True)
df_test = pd.read_csv(test_data, skipinitialspace=True)
df_all = pd.concat([df_train, df_test], ignore_index=True)

########################################
########GENERAL VARIABLES
########################################
vocab_length=len(set(df_all['lex'].values.tolist()))

dimensions=300
tr_epochs=10
tr_dropout=.4
embed_bipass_dims=20

N_CLASSES=87
hdn_layers=[100, N_CLASSES]


########################################
########CLASSES & FUNCTIONS
########################################

class embeds:

        ##If you're starting off with having generated word embeddings,
        # call this guy first.
        def get_embeds(dfk, mod=d2vs, input_columns=['lex']):
                #Creates a single list of lexical items from the summ total of
                # lexical units being passed to the network . . .
                lexemes=[]
                for col in input_columns:
                        lexemes+=dfk[col].values.tolist()

                #Final list of vectors to be used in the LSTM
                vec_vocab=[0 for word in set(lexemes)]

                #this is necessary for conversion of items in batch_sents(),
                # where each word will be replaced with its vector rep in the
                # network. We start by creating an id#-to-word dictionary, and
                # then just flip it to have a word-to id#, where the ID# is the
                # index of the word in the list of vectors.
                word2id = {}
                ct=0
                for word in set(lexemes):
                        word2id[word]=ct
                        ct+=1

                #Here we take the empty list vec_vocab, and fill it with vectors
                # for our document vocab. Also collects info about KeyErrors...
                errors=0
                for word, idx in word2id.items():
                        try:
                                vec_vocab[word2id[word]]=mod.wv[lem.lemmatize(str(word))]
                        except KeyError:
                                vec_vocab[word2id[word]]=np.array([0.0 for k in range(dimensions)])
                                if word!=0:
                                        errors+=1

                #Finally, we return the list of vectors (a list of np.arrays) and
                # our conversion dictionary, mapping words to vec_vocab indeces,
                # which are then used in the LSTM batches.
                print(errors, ' vocabulary items not translated into vecs.')
                return word2id, np.array(vec_vocab)                                       
                                

        def check_embeds(dfk, mod=d2vs):
                #A simple algorithm to check for errors, this will check if the
                # lexeme is in the dictionary, and if it is not it will then
                # count it as an error. The function then prints the percentage
                # of KeyErrors in the data being passed to it. This is useful in
                # ensuring how much of the system will be populated with blank
                # np.arrays ([0.0*dims]).
                errors=[]
                
                for it in set(dfk['lex'].values.tolist()):
                        try:
                                mod.wv[it]
                        except KeyError:
                                errors.append(it)

                print(len(errors), ' # words not in dictionary.\n')
                return errors
                                

                

class net:
                
        def lstm(x, y, word2iddict, embeddings=[], drop=tr_dropout, hidden_layers=hdn_layers, epochs=tr_epochs, embed_dims=dimensions, dfk=df_train):
                #Creates a model Sequential() instance and establishes the input
                # layer as the embeddings. We lastly add the LSTM Hidden layer,
                # and for shiggles populate it with 300 hidden units.
                model=Sequential()

                #This switch either loads embeddings if you have them, or
                # initializes the inputs with a trainable embedding layer.
                if embeddings.any():
                        model.add(Embedding(len(word2iddict), embed_dims, weights=[embeddings]))
                else:
                        model.add(Embedding(len(word2iddict), embed_bipass_dims))
                
                ##It took a minute, but I realized that the LSTM layer requires
                # defining the input-shape. Go figure.
                model.add(LSTM(300, return_sequences=True, activation='softmax'))
                model.add(Dropout(drop))
                for layer in hidden_layers:
                        model.add(Dense(layer, activation='relu'))
                        model.add(Dropout(drop))
                
                model.add(Dense(N_CLASSES, activation='softmax'))
                model.compile('rmsprop', 'categorical_crossentropy')

                ###INPUT-FN
                #While there may be some useful functions in any of the libraries
                # I'm using, I ended up having to convert my labels to a 1-hot vec
                # via brute force. To do this, I initialized a 1-hot shell list,
                # where all values were zero, with a length of the number of classes
                # being calced. This is then cloned and changed later at step LABEL
                # below.
                lhot=[0 for i in range(N_CLASSES)]
                ####BY BATCH
                #Rather than training everything at a go, I split it all into batches
                # a priori. Per each batch, it goes through all sentences and also
                # prints the batch #.
                for i in range(epochs):
                        print('Training epoch {}'.format(i))

                        for n_sent in set(dfk['sent'].values.tolist()):
                                s=dfk.loc[dfk['sent'].isin([n_sent])]

                                ##To create the labels, I cloned the list lhot, and then
                                # at the appropriate axis as inicated by the label #
                                # changed the 0 to a 1. This was then copied to the list
                                # label and then converted to an np array.
                                ##The array needed to have dimensions that matched (1) the
                                # batch size as derived from s, (2) the number of rows, &
                                # (3) the number of columns. It may be, that len(s) and
                                # -1 need be switched . . . I'm not sure here.
                                label=[]
                                for l in s['Labels'].values.tolist():
                                        a=list(lhot)
                                        a[l]=1
                                        label.append(a)
                                label=np.array(label).reshape((len(s), -1, N_CLASSES))

                                ##This one was tricker than expected . . . so the embeddings
                                # layer does not take an embedding, but rather wants the
                                # dictionary item # . . . screw the word, it wants the numeric
                                # hash-bucket rep. This actually makes sense for tf. So, we
                                # create an array of all the hash-bucket items, and pass that
                                # as inputs. Easy-peasy.
                                sent=[]
                                for word in s['lex'].values.tolist():
                                        sent.append(word2iddict[lem.lemmatize(word)])
                                sent=np.array(sent).reshape( -1, 1)

                                ##Annnd, train on batch.
                                model.train_on_batch(sent, label)

                return model


        
        ##########################################
        ######### LSTM PROCESS BROKEN UP
        ##########################################
        def lstm_model(word2iddict, embeddings=[], drop=tr_dropout, hidden_layers=hdn_layers, embed_dims=dimensions):
                model=Sequential()
                
                if embeddings.any():
                        model.add(Embedding(len(word2iddict), embed_dims, weights=[embeddings]))
                else:
                        model.add(Embedding(len(word2iddict), embed_bipass_dims))
                
                model.add(LSTM(300, return_sequences=True, activation='softmax'))
                model.add(Dropout(drop))

                for layer in hidden_layers:
                        model.add(Dense(layer, activation='relu'))
                        model.add(Dropout(drop))
                
                model.add(Dense(N_CLASSES, activation='softmax'))
                model.compile(optimizer='rmsprop', loss= 'categorical_crossentropy', metrics=['accuracy'])

                return model
        
        def lstm_model_functional(word2iddict, embeddings=[], drop=tr_dropout, hidden_layers=hdn_layers, embed_dims=dimensions):
                x1 = Input(shape=( None, ))

                net = Embedding(len(word2iddict), embed_dims, weights=[embeddings])(x1)
                net = LSTM(300, return_sequences=True, activation='softmax', name='lstm')(net)
                net = Dropout(drop)(net)
                for layer in hidden_layers:
                        net = Dense(layer, activation='relu')(net)
                        net = Dropout(drop)(net)

                predictions = Dense(N_CLASSES, activation='softmax')(net)

                model = Model(inputs=x1, outputs=predictions)
                model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

                return model

        def input_fn(dfk, word2iddict):
                x=[]
                y=[]
                ###INPUT-FN
                #While there may be some useful functions in any of the libraries
                # I'm using, I ended up having to convert my labels to a 1-hot vec
                # via brute force. To do this, I initialized a 1-hot shell list,
                # where all values were zero, with a length of the number of classes
                # being calced. This is then cloned and changed later at step LABEL
                # below.
                lhot=[0 for i in range(N_CLASSES)]
                
                for n_sent in set(dfk['sent'].values.tolist()):
                        s=dfk.loc[dfk['sent'].isin([n_sent])]

                        label=[]
                        for l in s['Labels'].values.tolist():
                                a=list(lhot)
                                a[l]=1
                                label.append(a)
                        label=np.array(label).reshape(( len(s), -1,  N_CLASSES))
                        y.append(label)

                        sent=[]
                        for word in s['lex'].values.tolist():
                                sent.append(word2iddict[lem.lemmatize(word)])
                        sent=np.array(sent).reshape( -1, 1)
                        x.append(sent)

                return x, y

        def train_on_batch(x, y, NN, epochs=tr_epochs):
                training=list(zip(x,y))

                for i in range(epochs):
                        print('Training Epoch {}'.format(i))

                        for it in training:
                                NN.train_on_batch(it[0], it[1])
                                

        def evaluate(x, y, NN):
                correct=0
                #true_labels=y
                acc=[]
                loss=[]

                ct=0
                for it in x:
                        a = NN.evaluate(it, y[ct])
                        acc.append(a[1])
                        loss.append(a[0])
                        ct+=1

                avg_acc=sum(acc)/len(acc)
                avg_loss=sum(loss)/len(loss)

                print('@accF1: ', avg_acc)
                print('@loss: ', avg_loss)
                return avg_acc, avg_loss

        
########################################
########IMPLEMENTATION
########################################
#errant=embeds.check_embeds(df_all)
#LSTM=net.lstm(['lex'], 'Labels', w2id, vectors)

w2id, vectors = embeds.get_embeds(df_all)
x, y = net.input_fn(df_train, w2id)
LSTM = net.lstm_model(w2id, vectors)
net.train_on_batch(x, y, LSTM)

x_pred, y_pred = net.input_fn(df_test, w2id)
correct = net.evaluate(x_pred, y_pred, LSTM)

