from typing import List
import json
import glob
import os
import re
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import BertTokenizer
from textblob import TextBlob

import subprocess


class IdfComputer:
    def __init__(self,
                 IDF_path: Path,
                 bert_model_name: str ='bert-base-cased'):
        """
        """
        self.IDF_path = IDF_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    @staticmethod
    def tokenizer_wrapper(text, tokenizer):
        if type(tokenizer) == BertTokenizer:
            tokenized_text = tokenizer.tokenize("[CLS] " + text + " [SEP]")
            indices = tokenizer.convert_tokens_to_ids(tokenized_text)
            return tokenized_text, indices
        else:
            print("Expecting a BertTokenizer")

    @staticmethod
    def detokenizer_wrapper(ids, tokenizer):
        if type(tokenizer) == BertTokenizer:
            return tokenizer.convert_ids_to_tokens([int(ids)])
        else:
            print("Expecting a BertTokenizer")

    @staticmethod
    def dummy_tokenizer(doc):
        return doc

    def get_idf(self, corpus):
        """
        Compute IDF values for a single corpus (list of sentences from selection of files).

        :param corpus: A single corpus (list of sentences)
        :return: Dict with IDF weights for all tokens found in the corpus
        """
        vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            use_idf=True,
            norm=None,
            smooth_idf=True,
            sublinear_tf=False,
            binary=False,
            # min_df=1, max_df=1.0, max_features=None, ngram_range=(1, 1),
            stop_words=None,
            analyzer='word',
            tokenizer=self.dummy_tokenizer,
            lowercase=False,
            preprocessor=self.dummy_tokenizer, vocabulary=None
        )
        vectorizer.fit_transform(corpus)
        idf_Y = vectorizer.idf_
        test_Y = dict(zip([str(x) for x in vectorizer.get_feature_names()], idf_Y))

        return test_Y

    def process_list_of_texts(self, list_of_texts):
        processed_list = []
        for sent in list_of_texts:
            tokens, indices = self.tokenizer_wrapper(sent, self.tokenizer)
            processed_list.append([str(vocab_idx) for vocab_idx in indices])

        if processed_list:
            return processed_list

    def compute_or_load_IDF_weights(self, corpus_list : List[str], overwrite=True):
        """
        Overarching function to compute or load the IDF weights, as well as train or load a SentencePiece model.
        :returns IDF_path: Path object
        """

        if self.IDF_path.exists() and not overwrite:   
            print("Loading existing IDF weights.")       # this part is deprecated
            with open(self.IDF_path, 'r') as f:
                IDF = json.load(f)
        else:
            print("Computing IDF weights.")
            processed_corpus_list = self.process_list_of_texts(corpus_list)
            IDF = self.get_idf(processed_corpus_list)
            with open(self.IDF_path, 'w') as f:
                json.dump(IDF, f)

        # print some tokens and IDF values to see what kind of stuff we get
        print("Printing some IDF values, should be subword units!")
        sanity_check = [x for x in IDF.keys()]
        for x in sanity_check[:10]:
            print(self.detokenizer_wrapper(x, self.tokenizer))

        return self.IDF_path
