import random
import pandas as pd
import numpy as np
import gensim
from nltk.stem import WordNetLemmatizer

import keras
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Bidirectional, Activation, Input, Concatenate, Reshape, add, maximum
from keras.layers import dot, multiply, MaxPooling1D, TimeDistributed
from keras.callbacks import TensorBoard

from keras_contrib.layers import  CRF

####################################################################################
######## To run . . .
####################################################################################
#nn.fit(seq, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, validation_data=(seq_, y_))
#acc, loss =batch.batched_training(df_train, ['lex'], ['n75-MOB'], nn, test_data=df_test)


####################################################################################
######## General variables
####################################################################################
BATCH_SHAPE=(None,1)
BATCH_SIZE=10
dimensions=300
EPOCHS=3
tr_dropout=.2
embed_bipass_dims=20
hdn_layers=[300, 150] #[1512, 756, 328]#[100, N_CLASSES]




####################################################################################
######## Data-sets
####################################################################################
data_idx= 0
train_datasets =  ['./1-data/exp-data/df_train20-argstr-MOBn75.csv',
                   './1-data/exp-data/df_train20-argstr-MOBn20.csv',
                   './1-data/exp-data/df_train20-argstr-splitless.csv',
                   './1-data/exp-data/v18/df_train18-HDBSCANn75-splitless-add.csv',
                   './1-data/exp-data/v18/df_train18-HDBSCANn75-splitless-flat.csv',
                  ]

test_datasets = ['./1-data/exp-data/df_test20-argstr-MOBn75.csv',
                 './1-data/exp-data/df_test20-argstr-MOBn20.csv',
                 './1-data/exp-data/df_test20-argstr-splitless.csv',
                 './1-data/exp-data/v18/df_test18-HDBSCANn75-splitless-add.csv',
                 './1-data/exp-data/v18/df_test18-HDBSCANn75-splitless-flat.csv',
                    ]

train_data=train_datasets[data_idx]
test_data=test_datasets[data_idx]

df_train = pd.read_csv(train_data, sep=',', skipinitialspace=True)
df_test = pd.read_csv(test_data, sep=',', skipinitialspace=True)
df_all = pd.DataFrame(np.array(df_train.values.tolist() + df_test.values.tolist()).reshape(-1, len(list(df_train))), columns=list(df_train))

acc_out='./1-data/acc.csv'

####################################################################################
######## Derived variables
####################################################################################
N_CLASSES=max(set(df_train['n20-MOB'].values.astype(int).tolist()))+1
test_classes=max(set(df_test['n75-MOB'].values.tolist()))


####################################################################################
######## Lemmatization
####################################################################################
lem=WordNetLemmatizer()


####################################################################################
######## Word vecs
####################################################################################
a2v_model_path='./1-data/_vec-models/mohler-d2v-MOB.bin'

d2vs=gensim.models.Word2Vec.load(a2v_model_path)
                                


####################################################################################
####### Embeddings
 ####################################################################################
class embeds:

        ##If you're starting off with having generated word embeddings,
        # call this guy first.
        def get_embeds(dfk, input_columns=['tref', 'lex'], fx='Normal', mod=d2vs):
                #Creates a single list of lexical items from the summ total of
                # lexical units being passed to the network . . .
                lexemes=[]
                for col in input_columns:
                        for word in dfk[col].values.tolist():
                                if fx=='split':
                                        MWE=word.replace('1', '').split('_')
                                        for lex in MWE:
                                                lexemes.append(lem.lemmatize(str(lex)))
                                if fx=='Normal':
                                        lexemes.append(lem.lemmatize(str(word)))

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
                errors=[]
                for word, idx in word2id.items():
                        try:
                                vec_vocab[word2id[word]]=mod.wv[str(word)]
                        except KeyError:
                                vec_vocab[word2id[str(word)]]=np.random.rand(dimensions)
                                if word!=str(0):
                                        errors.append(str(word))

                #Finally, we return the list of vectors (a list of np.arrays) and
                # our conversion dictionary, mapping words to vec_vocab indeces,
                # which are then used in the LSTM batches.
                print(len(set(errors)), ' vocabulary items not translated into vecs.')
                return word2id, np.array(vec_vocab) , errors

w2id, vectors, mal = embeds.get_embeds(df_all, ['lex']) #['lex']# #['tref', 'nsubj', 'dobj', 'verb', 'iobj', 'obl1', 'obl2']
vocab_length=len(vectors)


####################################################################################
####### Inputs & outputs
 ####################################################################################

#######################
### Inputs/Features
#######################
class inputs:
        
        def variableHashColumn(col, dfk, sent_id_col='sent', word2iddict=w2id):
                x=[]
                ###INPUT-FN
                #This implementation presumes that you have a set of inputs
                # with variable length in your dataset. This Fx will create a list of
                # batches that the model can be trained on iteratively, and pre-
                # supposes that you intend to use the train_on_batch() function
                # in Keras
                
                for n_sent in sorted(set(dfk[sent_id_col].values.tolist())):
                        s=dfk.loc[dfk[sent_id_col].isin([n_sent])]

                        sent=[]
                        for word in s[col].values.tolist():
                                sent.append(word2iddict[lem.lemmatize(str(word))])#str(word)])#
                        sent=np.array(sent).reshape( -1, 1)
                        x.append(sent)

                return x
        
        def hashColumn(column, dfk, vecs=vectors, dic=w2id, fx=None):
                data=[]

                for word in dfk[column].values.tolist():
                        lex=''
                        if fx==None:
                                lex=lem.lemmatize(str(word))
                        if fx=='Flat':
                                lex=lem.lemmatize(str(word).replace('1', '').split('_')[-1])
                        if fx=='Stanford':
                                lex=lem.lemmatize(
                                        head.stanford(str(word).replace('1', '').replace('_', ' '))
                                        )
                        data.append(dic[lex])#vecs[dic[lex]]

                return np.array(data).reshape(-1,1)#.reshape(-1, 300)


#######################
### Outputs/Lables
#######################
class output:

        def labels(labelish, dfk):
                labels=[]
                
                one_hot=[0 for i in range(N_CLASSES)]
                for l in dfk[labelish].values.tolist():
                        a=list(one_hot)
                        a[l]=1
                        labels.append(a)

                return np.array(labels).reshape(-1, 1,
                                                N_CLASSES)


        def variableLabels(col, dfk, sent_id_col='sent', word2iddict=w2id):
                y=[]
                ###INPUT-FN
                #This implementation presumes that you have a set of inputs
                # with variable length in your dataset. This Fx will create a list of
                # batches that the model can be trained on iteratively, and pre-
                # supposes that you intend to use the train_on_batch() function
                # in Keras

                oneHot=[0.0 for i in range(N_CLASSES)]
                
                for n_sent in sorted(set(dfk[sent_id_col].values.tolist())):
                        s=dfk.loc[dfk[sent_id_col].isin([n_sent])]

                        sent=[]
                        for l in s[col].values.tolist():
                                a=list(oneHot)
                                a[l] = 1.0
                                sent.append(a)
                        sent=np.array(sent).reshape( -1, 1,  N_CLASSES)
                        y.append(sent)

                return y
             

        def binaryLabels(col, dfk, mode=None, sent_id_col='sent', word2iddict=w2id):
                y=[]
                oneHot=[0.0, 0.0]
                maximum=max(set(dfk[col].values.tolist()))

                for n_sent in sorted(set(dfk[sent_id_col].values.tolist())):
                        s=dfk.loc[dfk[sent_id_col].isin([n_sent])]

                        sent=[]
                        for l in s[col].values.tolist():
                                a=list(oneHot)
                                if l != maximum:
                                        a[1] = 1.0
                                else:
                                        a[0] = 1.0
                                sent.append(a)
                        sent=np.array(sent).reshape( -1, 1, 2)
                        y.append(sent)

                if mode=='variable':
                        return y
                else:
                        return np.array(y).reshape(-1, 1, 2)
                                        
        
####################################################################################
####### Neural Network
####################################################################################
class net:

        def stackedLSTM(embeddings=[], word2iddict=w2id, drop=tr_dropout, hidden_layers=hdn_layers, embed_dims=dimensions):
                seq_input = Input(batch_shape=BATCH_SHAPE)
                seq = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, weights=[embeddings],
                                name='seq')(seq_input)

                kGEN = Bidirectional(LSTM(dimensions, return_sequences=True, activation='relu'))(seq)
                kGEN = LSTM(N_CLASSES, return_sequences=True, activation='softmax')(kGEN)

                kGEN = TimeDistributed(Dense(N_CLASSES, activation='relu'))(kGEN)
                crf = CRF(N_CLASSES)
                seq_out = crf(kGEN)
                
                model = Model(inputs=seq_input, outputs=seq_out)
                model.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])
                print(model.metrics_names)
                return model


class batch:
                
        def eval_on_batch(dfk, x, y, NN):
                acc=[]
                loss=[]
                sentID=[]

                for sent in set(dfk['sent'].values.tolist()):
                        s=dfk.loc[dfk['sent'].isin([sent])]
                        
                        x_vals=[]
                        for col in x:
                                x_vals.append(inputs.hashColumn(col, s))
                        y_vals=[]
                        for col in y:
                                y_vals.append(output.labels(col, s))
                                
                        los, ac = NN.evaluate(x_vals, y_vals, verbose=0)
                        acc.append(ac)
                        loss.append(los)
                        sentID.append(sent)

                print('@accF1: {} <---> @loss: {} \n'.format(sum(acc)/len(acc), sum(loss)/len(loss)))
                return acc, loss, sentID

        def batched_training(train_data, x_in, y_in, NN, test_data, ep=EPOCHS):
                acc_all=[]
                loss_all=[]
                
                for i in range(ep):
                        print('Epoch {}/{}'.format(i+1, ep))
                        loss=[]
                        acc=[]
                        for sent in set(train_data['sent'].sample(frac=1).values.tolist()):
                                s=train_data.loc[train_data['sent'].isin([sent])]

                                x_vals=[]
                                y_vals=[]
                                for col in x_in:
                                        x_vals.append(inputs.hashColumn(col, s))
                                for col in y_in:
                                        y_vals.append(output.labels(col, s))

                                los, ac = NN.train_on_batch(x_vals, y_vals)
                                loss.append(los)
                                acc.append(acc)

                        if test_data.empty == False:
                                stats=batch.eval_on_batch(test_data, x_in, y_in, NN)
                                acc=stats[0]
                                loss=stats[1]

                        acc_all.append(acc)
                        loss_all.append(loss)

                return acc_all, loss_all

       
####################################################################################
####### Implementation
####################################################################################
print('{} classes in training data.'.format(N_CLASSES))
print('{} number of training steps to be run per epoch.'.format(len(df_train)))
print('{} test steps per validation.'.format(len(df_test)))
#print('len(train)%6:   {}     . . . tests if MOB or not retroactively.'.format(len(df_train)%BATCH_SIZE))
cont=input('continue? ')

#LSTM=lstm.simple(vectors)
nn=net.stackedLSTM(vectors)
acc, loss =batch.batched_training(df_train, ['lex'], ['n75-MOB'], nn, test_data=df_test)


