from typing import List, Union
import sys
import os
import imp
import requests

from tqdm import tqdm
from pathlib import Path
from textblob import TextBlob

from utilities.customdocument import CustomDocument


def split_list(some_list: List, chunk_size: int=8) -> List[List]:
    """
    Helper function to split a list into smaller lists of a given size.

    :param some_list:   List that has to be split into chunks.
    :param chunk_size:  Size of the sublists that will be returned.
    :return list_of_sublists:  A list of sublists, each with a maximum size of `chunk_size`.
    """
    return [some_list[i:i + chunk_size] for i in range(0, len(some_list), chunk_size)]


class NER: 
    def __init__(self, 
                 api: str = "http://localhost:8501/", 
                 split_length: int = 300):
        """
        
        """
        self.api = api
        self.split_length = split_length   # in number of tokens
        
    def set_number_of_instances(num_threads: int = 4):
        response = requests.post(self.api + "set_number_of_predictors/", num_threads)
        print(f"The number of instances was set to {num_threads}; {response}")

    def split_into_sentences(self, to_be_split: Union[str, List[str]]) -> List[str]:
        """
        There is no need to call this function, text will be split into sentences inside the container as well!
        """
        if type(to_be_split) == str:
            if ';' in to_be_split:
                # some of the WikiData definitions contain multiple definitions separated by ';'
                to_be_split = to_be_split.split(';')
            else:
                to_be_split = [to_be_split]
            
        sentences = []
        for text in to_be_split:
            for part in text.split('\n'):
                # split into sentences using PunktSentTokenizer (TextBlob implements NLTK's version under the hood) 
                sentences += [str(s) for s in TextBlob(part.strip()).sentences if len(str(s)) > 10]
        return sentences
    
    def process_text(self, text: Union[str, List[str]]):
        """
        This is the actual call to the SPaR.txt API.
        """
        response = requests.post(self.api + "predict_objects/",  json={"texts": text}).json()
        # texts = response['texts']
        sentences = response['sentences']
        predictions = response['predictions']
        
        return sentences, predictions
    
    def process_custom_document(self, input_document: CustomDocument):
        """
        """
        print(f"Working on: {input_document.source_fp}")
        content_as_list_of_dicts = input_document.to_list_of_dicts()
        idx_to_be_processed = [i for i, c in enumerate(content_as_list_of_dicts) if not c["meta"]["SPaR_labels"]]
            
        # determine which texts have to be processed
        texts_to_be_processed = []
        for content_idx, content_dict in enumerate(content_as_list_of_dicts):
            if content_idx not in idx_to_be_processed:
                continue
            
            # some preprocessing of texts
            text = ' '.join([x for x in content_dict["content"].split(' ') if x != '' and len(x) > 10])
            # some really long paragraphs in the EU regulations are summations that should be split at ';'
            if len(text) > 3000:
                text = text.replace(";", ".\n")
            if not text:
                continue
            
            texts_to_be_processed.append((content_idx, text))
        
        # Here we actually process the texts, in subsets of 80 texts so we can show a progress bar
        for subset in tqdm(split_list(texts_to_be_processed, 80)):
            indices, texts_list = zip(*subset)
            sentences_lists, subset_predictions_list = self.process_text(texts_list)
            
            # flatten objects per subset_text
            object_lists = [[o for p in p_list for o in p['obj']] for p_list in subset_predictions_list]
            
            # Store sentences and predicted labels (only care about the objects for now)
            idx_sents_objects = list(zip(indices, sentences_lists, object_lists))
            for idx, sentences, objects in idx_sents_objects:
                content_as_list_of_dicts[idx]["meta"]["sentences"] = '###'.join(sentences)   
                content_as_list_of_dicts[idx]["meta"]["SPaR_labels"] = ', '.join(objects)
            
            # Each processed subset is written to file. This way we can 
            # resume to the last subset we were working on.
            input_document.replace_contents(content_as_list_of_dicts)
            input_document.write_document()
        
        total_number_of_sentences_found = sum([len(c['meta']['sentences'].split('###')) for c in content_as_list_of_dicts])
        print(f"Number of sentences found: {total_number_of_sentences_found}")
        input_document.replace_contents(content_as_list_of_dicts)
        input_document.write_document()
        
        