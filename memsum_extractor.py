import json, os, sys
from my_summarizers import ExtractiveSummarizer_MemSum_Final

import nltk
from nltk.tokenize import sent_tokenize

memsum_model = ExtractiveSummarizer_MemSum_Final( 
             "model/MemSum_Final/pubmed_full/200dim/final/model.pt",
             "model/glove/vocabulary_200dim.pkl",  
             gpu = 0,
             embed_dim = 200,
             max_doc_len  = 500,
             max_seq_len = 100
             )

load_path = "../paraphrase/test_corpora/archive/"
save_path = "../paraphrase/test_corpora/extracted_archive/"
list_of_files = sorted(os.listdir(load_path))

file_path = sys.argv[1]
file_name = file_path.split("/")[-1]
print(file_path, file_name)

if file_name.endswith(".txt"):
    full_file = load_path+file_name
    if os.stat(full_file).st_size != 0:
        text = open(full_file)
        # print(text.read())
        list_of_sentences = sent_tokenize(text.read())
        list_of_sentences = [x.replace("\n"," ") for x in list_of_sentences]
        print(list_of_sentences[0:5])
        extracted_summary = memsum_model.extract([list_of_sentences],
                                                p_stop_thres=0.6,
                                                max_extracted_sentences_per_document= 7, 
                                                return_sentence_position= False )[0]
        print(extracted_summary)
        text.close()
        savefile = open(save_path+"extracted_"+file_name, "w")
        savefile.write("\n".join(item for item in extracted_summary))
        savefile.close()
    