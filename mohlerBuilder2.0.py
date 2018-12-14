import pandas as pd
import numpy as np
import time
import gensim
from random import randint
import xml.etree.ElementTree as ET
from nltk.stem import WordNetLemmatizer
from nltk.parse.stanford import StanfordDependencyParser as sparse
pathmodelsjar = '/Users/ZaqRosen/nltk_data/stanford-english-corenlp-2016-01-10-models.jar'
pathjar = '/Users/ZaqRosen/nltk_data/stanford-parser/stanford-parser.jar'
depparse = sparse(path_to_jar=pathjar, path_to_models_jar=pathmodelsjar)

tree = ET.parse('/Users/ZaqRosen/Documents/Corpora/mohler/en_small.xml')
root = tree.getroot()


####################################################################################
############################## In and out data
####################################################################################
out_data_sheet='./0-unprocessed_data/all_data-CoNLL-Structure-v2.csv'
in_data='./0-unprocessed_data/all_data-CoNLL-Structure-v1.csv'
train_out='./1-data/exp-data/df_train20-argstr-MOBn20.csv'
test_out='./1-data/exp-data/df_test20-argstr-MOBn20.csv'

df_sentsToLabels = pd.read_csv('./1-data/_labels-dictionary/all_data14-sentToTag_MET.csv', sep=',', skipinitialspace=True)
df_n75 = pd.read_csv('./1-data/_labels-dictionary/HDBSCAN-n75-dic.csv', sep=',', skipinitialspace=True)
df_n20 = pd.read_csv('./1-data/_labels-dictionary/HDBSCAN-n20-dic.csv', sep=',', skipinitialspace=True)
df_in=pd.read_csv(in_data, skipinitialspace=True)



####################################################################################
######## Lemmatization
####################################################################################
lem=WordNetLemmatizer()


####################################################################################
############################## Dependency parser
####################################################################################
class deps:

        def stanford(sentence, searchterm, sentID):
                data=[]
                err={'OS': 0, 'AS': 0, 'Uni': 0}

                try:
                        res= depparse.raw_parse(sentence)
                        dep=res.__next__()
                        ventral_stream= list(dep.triples())
                        for tuple in ventral_stream:
                                data.append([searchterm, tuple[0][0], tuple[0][1], tuple[1], tuple[2][0], tuple[2][1], sentID])

                except OSError:
                        err['OS']+=1
                except AssertionError:
                        err['AS']+=1
                except UnicodeEncodeError:
                        err['Uni']+=1

                return data, err

        def test(sentence):
                res = depparse.raw_parse(sentence)
                dep = res.__next__()
                ventral_stream = list(dep.triples())
                for tuple in ventral_stream:
                        print(tuple)

####################################################################################
############################## XML reader
####################################################################################
class xml:
        
        def eyes(root=root, verbose=False):
                err={'OS': 0, 'AS': 0, 'Uni': 0}
                df=pd.DataFrame(np.array([0 for i in range(7)]).reshape(-1, 7), columns=['tref', 'headFn', 'headPOS', 'clauseEl', 'lex', 'lexPOS', 'sentID'])
                df.to_csv(out_data_sheet, index=False, encoding='utf-8')
                ct=0
                tot=len(root.findall('LmInstance'))
                for chi in root.findall('LmInstance'):
                        ct+=1
                        sent_bits=''
                        LMTarget=''
                        sentspot=int(chi.attrib['id'])
                        
                        text = chi.find('TextContent')
                        current=text.find('Current')

                        annotations=chi.find('Annotations')
                        CMSOURCE = annotations.find('CMSourceAnnotations')
                        if CMSOURCE is None:
                                continue
                        else:
                                for element in current:
                                        if element.tag=='LmTarget':
                                                LMTarget=str(element.text).replace(' ', '_')+'kGEN12'
                                                sent_bits+=' '+LMTarget+' '
                                        else:
                                                sent_bits+=str(element.text)+' '
                                
                                sententials, errors = deps.stanford(sent_bits, LMTarget, sentspot)
                                outbound=pd.DataFrame(np.array(sententials).reshape(-1, 7), columns=['tref', 'headFn', 'headPOS', 'clauseEl', 'lex', 'lexPOS', 'sentID'])
                                outbound.to_csv(out_data_sheet, header=False, index=False, mode='a', encoding='utf-8')
                                err['OS']+=errors['OS']
                                err['AS']+=errors['AS']
                                err['Uni']+=errors['Uni']
                                print('Sentence {} / {}'.format(ct, tot))
                                
                        if verbose != False:                               
                                print(sent_bits+'\n')
                                
                return pd.read_csv(out_data_sheet, skipinitialspace=True), err


####################################################################################
############################## Converting dependencies (concatenae) to CxG construction types
####################################################################################
class cxn:

        def mob_splitter(listin):
                data=[]

                for it in listin:
                        if it[0]==it[4]:
                                MWE=str(it[4]).replace('kGEN12', '').split('_')
                                for word in MWE:
                                        a=list(it)
                                        a[4]=word
                                        data.append(a)
                        else:
                                data.append(it)

                return data

        
        def head_functors(dfk, verbose=False):
                data=[]

                ct=0
                for sent in dfk['sentID'].unique():
                        s=dfk.loc[dfk['sentID'].isin([sent])]
                        tref=s['tref'].values.tolist()[0]
                        head=s['headFn'].loc[s['lex'].isin([tref])].values.tolist()
                        data+=s.loc[s['headFn'].isin(head)].values.tolist()

                        if verbose==True:
                                print('Sentence {} / {}'.format(ct, len(set(dfk['sentID'].values.tolist()))))
                        ct+=1

                return pd.DataFrame(np.array(data).reshape(-1, len(list(dfk))), columns=list(dfk)[:-1]+['sent'])


        def argument_structure(dfk, verbose=False, fx=None):
                data=[]
                err={'OS': 0, 'AS': 0, 'Uni': 0, 'verb-less': 0, 'JJ':0}

                ct=0
                for sent in set(dfk['sentID'].values.tolist()):

                        if verbose==True:
                                print('@ {} / {} -- SentID: {}'.format(ct, len(set(dfk['sentID'].values.tolist())), sent))
                        
                        s=dfk.loc[dfk['sentID'].isin([sent])]
                        tref=s['tref'].values.tolist()[0]
                        tref_row=s.loc[s['lex'].isin([tref])]

                        if len(tref_row) < 1:
                                err['verb-less']+=1
                        
                        elif 'subj' in tref_row['clauseEl'].values.tolist()[0]:
                                if fx=='MOB':
                                        data+=cxn.mob_splitter(s.loc[s['headFn'].isin(tref_row['headFn'].values.tolist())].values.tolist())
                                else:
                                        data += s.loc[s['headFn'].isin(tref_row['headFn'].values.tolist())].values.tolist()
                                                                     
                        elif 'dobj' in tref_row['clauseEl'].values.tolist()[0]:
                                if fx=='MOB':
                                        data+=cxn.mob_splitter(s.loc[s['headFn'].isin(tref_row['headFn'].values.tolist())].values.tolist())
                                else:
                                        data += s.loc[s['headFn'].isin(tref_row['headFn'].values.tolist())].values.tolist()

                        elif 'nmod' in tref_row['clauseEl'].values.tolist()[0]:
                                choices=s.loc[s['headFn'].isin([tref])].values.tolist()
                                for array in choices:
                                        if array[3]=='case':
                                                data+=[array]
                                if fx=='MOB':
                                        data+=cxn.mob_splitter(s.loc[s['headFn'].isin(tref_row['headFn'].values.tolist())].values.tolist())
                                else:
                                        data += s.loc[s['headFn'].isin(tref_row['headFn'].values.tolist())].values.tolist()
                                #print('mod')
                                
                        elif 'amod' in tref_row['clauseEl'].values.tolist()[0]:
                                lower_head=s['headFn'].loc[s['lex'].isin(tref_row['headFn'].values.tolist())].values.tolist()
                                upper_head=s['headFn'].loc[s['lex'].isin(lower_head)].values.tolist()
                                data+= s.loc[s['headFn'].isin(lower_head)].values.tolist()
                                choices=s.loc[s['headFn'].isin([tref])].values.tolist()
                                for array in choices:
                                        if array[3]=='case':
                                                data+=[array]
                                if fx=='MOB':
                                        data+=cxn.mob_splitter(tref_row.values.tolist())
                                else:
                                        data+=tref_row.values.tolist()
                                #print('mod')
                        
                        elif 'JJ' in tref_row['lexPOS'].values.tolist()[0]:
                                data+= s.loc[s['headFn'].isin(tref_row['headFn'].values.tolist())].values.tolist()
                                data+=tref_row.values.tolist()
                                err['JJ']+=1
                        
                        ct+=1
                        
                return pd.DataFrame(np.array(data).reshape(-1, len(list(dfk))), columns=list(dfk)[:-1]+['sent']), err



class data:

        def add_labels(dfi, old_label_name, new_label_name, dic):
                print('{} --> {}'.format(old_label_name, new_label_name))

                labels_dic={}
                for it in dic.values.tolist():
                        labels_dic[str(it[0])]=it[1]

                out=[]
                err=[]
                for loc in range(len(dfi)):
                        try:
                                out.append(dfi.loc[loc].values.tolist() + [labels_dic[str(dfi[old_label_name].loc[loc])]])
                        except KeyError:
                                err.append(dfi[old_label_name].loc[loc])

                print('{} sents not translated. \n'.format(len(set(err))))
                return pd.DataFrame(np.array(out).reshape(-1, len(list(dfi))+1), columns=list(dfi)+[new_label_name]), err

        def add_MOB_labels(dfi, main_label_name, new_label_name):
                null_label=max(set(dfi[main_label_name].values.astype(int).tolist()))+1

                new_labels=[]
                for sent in dfi['sent'].unique():
                        s=dfi.loc[dfi['sent'].isin([sent])]
                        label=s[main_label_name].values.astype(int).tolist()[0]
                        for it in s[['tref', 'lex']].values.tolist():
                                if it[0]==it[1]:
                                        new_labels.append(label)
                                else:
                                        new_labels.append(null_label)

                dfi[new_label_name]=new_labels

                return dfi
                        
                        

        def train_test(dfk, labels=None, pct=.8):
                train_sents=[]
                test_sents=[]

                for label in list(set(dfk[labels].values.tolist())):
                        a = dfk['sent'].loc[dfk[labels].isin([label])].tolist()
                        alist = list(set(a))
                        length=len(alist)
                        for k in range(int(length*pct)):
                                randIDX = randint(0, int(len(alist)-1))
                                train_sents.append(alist[randIDX])
                                alist.remove(alist[randIDX])
                        test_sents+=alist

                train=dfk.loc[dfk['sent'].isin(train_sents)].values.tolist()
                test=dfk.loc[dfk['sent'].isin(test_sents)].values.tolist()
                return pd.DataFrame(np.array(train).reshape(-1, len(list(dfk))), columns=list(dfk)), pd.DataFrame(np.array(test).reshape(-1, len(list(dfk))), columns=list(dfk))


class w2v:

        def corpus(col, dfk, lemmatization=True):
                tok_corpus=[]

                for sent in dfk['sent'].unique():
                        s=dfk[col].loc[dfk['sent'].isin([sent])].values.tolist()
                        if lemmatization==True:
                                s=[lem.lemmatize(str(w)) for w in s]
                        tok_corpus.append(s)

                return tok_corpus

        def model(corpus, dims=300, minimum=1):
                model = gensim.models.Word2Vec(corpus, min_count=minimum, size=dims)
                return model

        def save_model(model, outfile):
                model.save(outfile)
        

####################################################################################
############################## Implementation
####################################################################################

dfArgs, errors = cxn.argument_structure(df_in, fx='MOB')
dfArgs, errors2 = data.add_labels(dfArgs, 'sent', 'main-labels', df_sentsToLabels)
dfArgs, errors3 = data.add_labels(dfArgs, 'main-labels', 'n75-labels', df_n75)
dfArgs, errors4 = data.add_labels(dfArgs, 'main-labels', 'n20-labels', df_n20)
dfArgs = data.add_MOB_labels(dfArgs, 'n75-labels', 'n75-MOB')
dfArgs = data.add_MOB_labels(dfArgs, 'n20-labels', 'n20-MOB')

drop=list(dfArgs.index[dfArgs['n20-labels'].isin([str(-1)])])
dfArgs=dfArgs.drop(drop)
dfArgs.index=range(len(dfArgs))

train, test = data.train_test(dfArgs, labels='n20-labels', pct=.85)
train.to_csv(train_out, index=False, encoding='utf-8')
test.to_csv(test_out, index=False, encoding='utf-8')

