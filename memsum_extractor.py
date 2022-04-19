import json, os
from my_summarizers import ExtractiveSummarizer_MemSum_Final

import nltk
from nltk.tokenize import sent_tokenize

memsum_model = ExtractiveSummarizer_MemSum_Final( 
             "model/MemSum_Final/pubmed_full/200dim/final/model.pt",
             "model/glove/vocabulary_200dim.pkl",  
             gpu = 4,
             embed_dim = 200,
             max_doc_len  = 500,
             max_seq_len = 100
             )

load_path = "../paraphrase/test_corpora/archive/"
list_of_files = sorted(os.listdir(load_path))

for file in list_of_files:
    if file.endswith(".txt"):
        text = open(load_path+file)
        # print(text.read())
        list_of_sentences = sent_tokenize(text.read())
        print(list_of_sentences[0:5])
        extracted_summary = memsum_model.extract(list_of_sentences,
                                                p_stop_thres=0.6,
                                                max_extracted_sentences_per_document= 7, 
                                                return_sentence_position= False )[0]
        print(extracted_summary)

        text.close()
        break