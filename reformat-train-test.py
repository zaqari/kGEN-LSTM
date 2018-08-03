import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer

########################################
########LEMMATIZATION SET-UP
########################################
lem=WordNetLemmatizer()


########################################
###############IMPORT TRAIN AND TEST
########################################
COLUMNS = ['LmTarget', 'subj', 'dobj', 'syn', 'verb',  'iobj',  'obl1', 'obl2', 'Labels']
convert_tr = './1-data/train_data14v4.csv'
convert_te = './1-data/test_data14v4.csv'
df_tr_in = pd.read_csv(convert_tr, skipinitialspace=True)
df_te_in = pd.read_csv(convert_te, skipinitialspace=True)


########################################
###############VARIABLES
########################################
NO_CLASS=int(len(set(df_tr_in['label'].values.tolist())))
train_out='./1-data/train_data14v5-mob.csv'
test_out='./1-data/test_data14v5-mob.csv'


########################################
###############CONVERSION TO LSTM FORMAT
########################################
class lstm_data:

        def columnized(dfk):
                data=[]

                ct=0
                for rep in dfk.values.tolist():
                        good_examples=list(rep[1:3])+list(rep[4:-1])
                        for word in good_examples:
                                if word==rep[0]:
                                        MWE=word.replace('1', '').split('_')
                                        #THE FOR LOOP here makes it +6, not (6,,)
                                        for it in MWE:
                                                data.append([lem.lemmatize(str(it)), ct, rep[-1]])
                                else:
                                        data.append([lem.lemmatize(str(word)), ct, NO_CLASS])
                        ct+=1

                return pd.DataFrame(np.array(data).reshape(-1, 3), columns=['lex', 'sent', 'Labels'])


########################################
###############CONVERSION TO LSTM FORMAT
########################################

print('Total n_categories: ', NO_CLASS+1)

######
##RE-BUILD DATA SETS
######
print('Reformatting training data!')
df_train=lstm_data.columnized(df_tr_in)
print('Reformatting test data!')
df_test=lstm_data.columnized(df_te_in)


######
##SAVE TO FILE
######
df_train.to_csv(train_out, sep=',', header=True, index=False, encoding='utf-8')
df_test.to_csv(test_out, sep=',', header=True, index=False, encoding='utf-8')
