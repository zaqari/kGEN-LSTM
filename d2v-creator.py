###################################
##########IMPORTS
###################################
import nltk
import pandas as pd
import numpy as np
import gensim

from nltk.parse.stanford import StanfordDependencyParser as sparse
pathmodelsjar = './nltk_data/stanford-english-corenlp-2016-01-10-models.jar'
pathjar = './nltk_data/stanford-parser/stanford-parser.jar'
depparse = sparse(path_to_jar=pathjar, path_to_models_jar=pathmodelsjar)

#################################
############VARIABLES
#################################
line_corpus='./corpora/texts/wikimedia.txt'
arg_data_sheet='./0-unprocessed_data/vec_data/argset-200k.csv'
d2v_data_sheet='./0-unprocessed_data/vec_data/d2vset-200k.csv'
nonce_word='77KellyIsaacs14'

d2v_outfile='./1-data/_vec-models/d2v-200k-1.bin'
a2v_outfile='./1-data/_vec-models/a2v-200k-1.bin'
###################################
##########DEP PARSER
###################################
class data:

        def lines_to_deps(corpus, cutoff=30):
                out_columns=[str(j) for j in range(cutoff)]
                cnt=0
                
                doc=open(corpus, encoding='utf-8')
                for line in doc:
                        mtrs=[]
                        try:
                                res=depparse.raw_parse(doc.readline())
                                dep=res.__next__()
                                ventral_stream=list(dep.triples())
                                for tuple in ventral_stream:
                                        mtrs.append(tuple[0])

                                for mom in set(mtrs):
                                        a=[mom[0]]
                                        for tuple in ventral_stream:
                                                if tuple[0]==mom:
                                                        a.append(tuple[2][0])

                                        if len(a) < cutoff:
                                                for k in range(len(a), cutoff):
                                                        a.append(nonce_word)

                                        if len(a) > cutoff:
                                                a=list(a[:cutoff])

                                        newdata=pd.DataFrame(np.array(a).reshape(-1, cutoff), columns=out_columns)

                                        if 'VB' in mom[1]:
                                                newdata.to_csv(d2v_data_sheet, sep=',', header=False, index=False, mode='a', encoding='utf-8')
                                                newdata.to_csv(arg_data_sheet, sep=',', header=False, index=False, mode='a', encoding='utf-8')
                                        else:
                                                newdata.to_csv(d2v_data_sheet, sep=',', header=False, index=False, mode='a', encoding='utf-8')

                                cnt+=1
                                print(cnt, ' sentences actually processed.')

                        except OSError:
                                print('OSError thrown.')
                        except AssertionError:
                                print('AssertionError Thrown')
                        except UnicodeEncodeError:
                                print('UnicodeError thrown')
                                        
                doc.close()

                return pd.read_csv(d2v_data_sheet, skipinitialspace=True), pd.read_csv(arg_data_sheet, skipinitialspace=True)


        def remove_nonces(dfk):
                tok_corpus=[]

                for item in dfk.values.tolist():
                        a=[]

                        for word in item:
                                if word!=nonce_word:
                                        a.append(str(word))

                        tok_corpus.append(a)

                return tok_corpus

        def hippocampus(data, outfiles, min_examples=5, dims=300):
                models=[]
                ct=0

                for lis in data:
                        model = gensim.models.Word2Vec(lis, min_count=min_examples, size=dims)
                        model.save(outfiles[ct])
                        models.append(model)
                        ct+=1

                return models


d2v_in=pd.read_csv(d2v_data_sheet, skipinitialspace=True)
a2v_in=pd.read_csv(arg_data_sheet, skipinitialspace=True)

d2vs=data.remove_nonces(d2v_in)
a2vs=data.remove_nonces(a2v_in)

models=data.hippocampus([d2vs, a2vs], [d2v_outfile, a2v_outfile], 1)
                        
