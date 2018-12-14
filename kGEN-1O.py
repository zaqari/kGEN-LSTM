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
d2v_model_path='./1-data/_vec-models/d2v-wiki200-wsj.bin'
a2v_model_path='./1-data/_vec-models/mohler-d2v-MOB.bin'

d2vs=gensim.models.Word2Vec.load(a2v_model_path)
#a2vs=gensim.models.Word2Vec.load(a2v_model_path)


####################################################################################
####### De facto analyses
####################################################################################
class analyses:

        def bestN(results, top_n=3, fx='max'):
                data=[]

                for row in range(len(results)):
                        if fx=='max':
                                data.append(list(np.argpartition(results[row], -3)[0][-1:-int(top_n+1):-1]))
                        if fx=='min':
                                data.append(list(np.argpartition(results[row], -3)[0][:int(top_n)]))

                return data


        def index_of_correct(true, topN_results, verbose=False):
                at_index=[]
                tru_label=[]
                check=list(zip(true, topN_results))

                for comp in check:
                        if comp[0] in comp[1]:
                                at_index.append(comp[1].index(comp[0]))
                                tru_label.append(comp[0])

                print('% @correct: {}'.format(len(at_index)/len(true)*100))
                if verbose==True:
                        print('@Index distribution for correct items')
                        for it in set(at_index):
                                print('% @index {} : {}'.format(it, at_index.count(it)/len(at_index)*100))

                return pd.DataFrame(np.array(list(zip(tru_label, at_index))).reshape(-1, 2), columns=['label', '@index'])
                                
                                
#######HOW TO#######

        
####################################################################################
####### Capture Head-Functor via dependencies
####################################################################################
class head:

        def stanford(phrase):
                head=''
                err_dic={"OSError": 0, "AssertionError":0,"UnicodeError":0}
                
                try:
                        res = depparse.raw_parse(phrase)
                        dep = res.__next__()
                        ventral_stream = list(dep.triples())

                        head_list=[]
                        for tuple in ventral_stream:
                                head_list.append(tuple[0])

                        head=lem.lemmatize(head_list[0][0])
                                
                except OSError:
                        err_dic['OSError']+=1
                except AssertionError:
                        err_dic['AssertionError']+=1
                except UnicodeError:
                        err_dic['UnicodeError']+=1

                return head


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
"""
lexical_cols=[]
if data_idx in [0,2]:
        lexical_cols=['tref', 'nsubj', 'dobj', 'verb', 'iobj', 'obl1', 'obl2']
else:
        lexical_cols=['lex']
"""
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


        def flatHash(dfk, columns=['verb', 'obl1', 'iobj', 'nsubj', 'dobj', 'obl2'], dic=w2id):
                x=[]

                for loc in range(len(dfk)):
                        data=[]
                        s=dfk[columns].loc[loc].values.tolist()
                        for word in s:
                                data.append(dic[lem.lemmatize(str(word))])
                        x.append(data)

                return np.array(x).reshape(-1, 1,
                                           len(columns))

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

        def flatLabels(col, dfk):
                y=[]
                oneHot=[0.0 for i in range(N_CLASSES)]
                for loc in range(len(dfk)):
                        s=dfk[col].loc[loc]
                        a=list(oneHot)
                        a[s]=1.0
                        y.append(a)
                return np.array(y).reshape(-1, 1,
                                           N_CLASSES)               

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

        def dnn(embeddings=[], word2iddict=w2id, drop=tr_dropout, hidden_layers=hdn_layers, embed_dims=dimensions):
                #Inputs = w2id number -> embedding layer as nn inputs
                seq_input = Input(batch_shape=BATCH_SHAPE)
                seq = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                name='seq')(seq_input)

                verb_input = Input(batch_shape=BATCH_SHAPE)
                verb = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                 name='verb')(verb_input)

                syn_input = Input(batch_shape=BATCH_SHAPE)
                syn = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                name='syn')(syn_input)

                subj_input = Input(batch_shape=BATCH_SHAPE)
                subj = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                 name='subj')(subj_input)

                dobj_input = Input(batch_shape=BATCH_SHAPE)
                dobj = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                 name='dobj')(dobj_input)

                obl1_input = Input(batch_shape=BATCH_SHAPE)
                obl1 = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                 name='obl1')(obl1_input)

                obl2_input = Input(batch_shape=BATCH_SHAPE)
                obl2 = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                 name='obl2')(obl2_input)

                iobj_input = Input(batch_shape=BATCH_SHAPE)
                iobj = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                 name='iobj')(iobj_input)

                seqxverbxsubj = add([seq, verb, subj])
                seqxverbxdobj = add([seq, verb, dobj])
                seqxverbxiobj = add([seq, verb, iobj])
                seqxobl1xobl2 = add([seq, obl1, obl2])

                kGEN = Concatenate()([seq, verb, syn, subj, dobj, iobj, obl1, obl2,
                                            seqxverbxsubj, seqxverbxdobj, seqxverbxiobj, seqxobl1xobl2])
                
                for layer in hidden_layers:
                        kGEN = Dense(layer, activation='relu')(kGEN)
                        kGEN = Dropout(drop)(kGEN)
                
                #Output & model compiling
                out = Dense(N_CLASSES, activation='softmax', name='out')(kGEN)
                
                model = Model(inputs=[seq_input, verb_input, syn_input, subj_input, dobj_input, obl1_input, obl2_input, iobj_input], outputs=[out])
                model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
                print(model.metrics_names)
                return model

        def simple(embeddings=[], word2iddict=w2id, drop=tr_dropout, hidden_layers=hdn_layers, embed_dims=dimensions):
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


        def flatLSTM(embeddings=[], word2iddict=w2id, drop=tr_dropout, hidden_layers=hdn_layers, embed_dims=dimensions):
                sample = Input(shape=BATCH_SHAPE)
                seq = Reshape(target_shape=(BATCH_SHAPE[1], ))(sample)
                seq = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, weights=[embeddings], name='seq')(seq)
                kGEN = Bidirectional(LSTM(600, activation='relu', return_sequences=True, dropout=drop, name='arg-str'), merge_mode='sum')(seq)

                #kGEN = Flatten()(seq)
                #kGEN = Reshape(target_shape=(BATCH_SHAPE[1], 600))(kGEN)
                #kGEN = MaxPooling1D(strides=6)(kGEN)
                
                for layer in hidden_layers:
                        kGEN = Dense(layer, activation='relu')(kGEN)
                        kGEN = Dropout(drop)(kGEN)
                
                out = TimeDistributed(Dense(N_CLASSES, activation='softmax', name='out'))(kGEN)
                
                model = Model(inputs=sample, outputs=out)
                model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
                print(model.metrics_names)
                return model

        def lstm2belief(embeddings=[], word2iddict=w2id, drop=tr_dropout, hidden_layers=hdn_layers, embed_dims=dimensions):
                #Inputs = w2id number -> embedding layer as nn inputs
                seq_input = Input(batch_shape=BATCH_SHAPE)
                seq = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                name='seq')(seq_input)

                verb_input = Input(batch_shape=BATCH_SHAPE)
                verb = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                 name='verb')(verb_input)

                syn_input = Input(batch_shape=BATCH_SHAPE)
                syn = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                name='syn')(syn_input)

                subj_input = Input(batch_shape=BATCH_SHAPE)
                subj = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                 name='subj')(subj_input)

                dobj_input = Input(batch_shape=BATCH_SHAPE)
                dobj = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                 name='dobj')(dobj_input)

                obl1_input = Input(batch_shape=BATCH_SHAPE)
                obl1 = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                 name='obl1')(obl1_input)

                obl2_input = Input(batch_shape=BATCH_SHAPE)
                obl2 = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                 name='obl2')(obl2_input)

                iobj_input = Input(batch_shape=BATCH_SHAPE)
                iobj = Embedding(input_dim=len(word2iddict), output_dim=embed_dims, #weights=[embeddings],
                                 name='iobj')(iobj_input)
                """
                seqxverbxsubj = add([seq, verb, subj])
                seqxverbxdobj = add([seq, verb, dobj])
                seqxverbxiobj = add([seq, verb, iobj])
                seqxobl1xobl2 = add([seq, obl1, obl2])
                """
                
                #LSTM formatting & model
                kGEN = Concatenate(axis=1)([seq, verb, syn, obl1, obl2, subj, dobj, iobj,
                                            #seqxverbxsubj, seqxverbxdobj, seqxverbxiobj, seqxobl1xobl2
                                            ])
                #kGEN = MaxPooling1D(strides=2)(kGEN)
                kGEN = Bidirectional(LSTM(300, activation='relu', return_sequences=True, dropout=drop, name='arg-str'), merge_mode='sum')(kGEN)
                kGEN = Bidirectional(LSTM(N_CLASSES, activation='softmax', return_sequences=True), merge_mode='sum')(kGEN)
                """
                #Deep Belief LSTM Component
                for layer in hidden_layers:
                        kGEN = LSTM(layer, activation='relu', return_sequences=True, dropout=drop)(kGEN)
                        kGEN = LSTM(N_CLASSES, activation='softmax', return_sequences=True)(kGEN)
                """
                kGEN = Flatten()(kGEN)
                #kGEN = MaxPooling1D(pool_size=8)(kGEN)

                #Deep Belief network with softmax gates.
                for layer in hidden_layers:
                        kGEN = Dense(layer, activation='relu')(kGEN)
                        kGEN = Dropout(drop)(kGEN)
                        kGEN = Dense(N_CLASSES, activation='softmax')(kGEN)

                #Output & model compiling
                out = Dense(N_CLASSES, activation='softmax', name='out')(kGEN)
                
                model = Model(inputs=[seq_input, verb_input, syn_input, subj_input, dobj_input, obl1_input, obl2_input, iobj_input], outputs=[out])
                model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
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
                        

        def pred_on_batch(f, NN, items=[]):
                acc=[]
                loss=[]

                evals=[]
                if items!=[]:
                        evals=items
                else:
                        evals=range(len(y))

                for i in evals:
                        ac = NN.predict([f[0][i], f[1][i], f[2][i], f[3][i], f[4][i], f[5][i], f[6][i]], verbose=0)
                        acc.append(ac)

                return acc

       
####################################################################################
####### Implementation
####################################################################################
print('{} classes in training data.'.format(N_CLASSES))
print('{} number of training steps to be run per epoch.'.format(len(df_train)))
print('{} test steps per validation.'.format(len(df_test)))
#print('len(train)%6:   {}     . . . tests if MOB or not retroactively.'.format(len(df_train)%BATCH_SIZE))
cont=input('continue? ')

inputables=['lex']#['tref', 'verb', 'syn', 'nsubj', 'dobj', 'obl1', 'obl2', 'iobj']
outputables=['Labels']

#seq, y = [inputs.hashColumn(col, df_train) for col in inputables], [output.labels(col, df_train) for col in outputables]
#seq_, y_ = [inputs.hashColumn(col, df_test) for col in inputables], [output.labels(col, df_test) for col in outputables]

#LSTM=lstm.simple(vectors)
nn=net.simple(vectors)
acc, loss =batch.batched_training(df_train, ['lex'], ['n75-MOB'], nn, test_data=df_test)
####NOTES####
#nn.fit(seq, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, validation_data=(seq_, y_))
#batch.batched_training(df_train, ['lex'], ['Labels'], LSTM, test_data=df_test)

"""

#########################################
######## Training
########################################
seq=inputs.hashColumn('lex', df_train)
syn=inputs.hashColumn('syn', df_train)
foc=inputs.hashColumn('tref', df_train)
verb=inputs.hashColumn('verb', df_train)
subj=inputs.hashColumn('nsubj', df_train)
dobj=inputs.hashColumn('dobj', df_train)
obl1=inputs.hashColumn('obl1', df_train)
obl2=inputs.hashColumn('obl2', df_train)

y=output.labels('Labels', df_train)
#y_alt=output.states('end-labels', df_train)
x=[seq, verb, syn, foc,
   #seq, foc, verb, subj, dobj, obl1, obl2
   ]

########################################
######## Testing
########################################
seq_=inputs.hashColumn('lex', df_test)
syn_=inputs.hashColumn('syn', df_test)
foc_=inputs.hashColumn('tref', df_test)
verb_=inputs.hashColumn('verb', df_test)
subj_=inputs.hashColumn('nsubj', df_test)
dobj_=inputs.hashColumn('dobj', df_test)
obl1_=inputs.hashColumn('obl1', df_test)
obl2_=inputs.hashColumn('obl2', df_test)

y_=output.labels('Labels', df_test)
#y_alt_=output.states('end-labels', df_test)
x_=[seq_, verb_, syn_, foc_,
    #seq_ foc_, verb_, subj_, dobj_, obl1_, obl2_
    ]
LSTM.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SHAPE[0], verbose=2, validation_data=(x_, y_))

####################################################################################
####### BATCHED TRAINING
####################################################################################

########################################
######## Training
########################################
seq=inputs.variableHashColumn('lex', df_train)
foc=inputs.variableHashColumn('tref', df_train)
verb=inputs.variableHashColumn('verb', df_train)
subj=inputs.variableHashColumn('nsubj', df_train)
dobj=inputs.variableHashColumn('dobj', df_train)
obl1=inputs.variableHashColumn('obl1', df_train)
obl2=inputs.variableHashColumn('obl2', df_train)
y=output.variableLabels('Labels', df_train)
x=[seq, foc, verb, subj, dobj, obl1, obl2]

########################################
######## Testing
########################################
seq_=inputs.variableHashColumn('lex', df_test)
foc_=inputs.variableHashColumn('tref', df_test)
verb_=inputs.variableHashColumn('verb', df_test)
subj_=inputs.variableHashColumn('nsubj', df_test)
dobj_=inputs.variableHashColumn('dobj', df_test)
obl1_=inputs.variableHashColumn('obl1', df_test)
obl2_=inputs.variableHashColumn('obl2', df_test)

x_=[seq_, foc_, verb_, subj_, dobj_, obl1_, obl2_]
y_=output.variableLabels('Labels', df_test)
lstm.on_batch(x, y, LSTM, x_, y_)
""" 
#y_dnn_=output.binaryLabels('main-labels', df_test)
#results=lstm.eval_on_batch(f_, y_, LSTM6)

