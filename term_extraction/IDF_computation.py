import json
import glob
import os
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import BertTokenizer
from textblob import TextBlob

import subprocess


class IdfComputer:
    def __init__(self,
                 IDF_path,
                 bert_model='whaleloops/phrase-bert',
                 conversion_type="pdf"):
        """
        CONSIDER MAKING THIS ITs OWN CONTAINER!
        """
        self.IDF_path = IDF_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

        if conversion_type in ["pdf", "ocr"]:
            self.input_file_dir = "/data/ir_data/pdf/"
        elif conversion_type == "xml":
            self.input_file_dir = "/data/ir_data/pdf/"       # todo; implement xml-based IDF, don't care now

    @staticmethod
    def read_pdf(file_path,
                 layout):
        """
        Extract pages from the pdf file at file_path.

        :param file_path: path of the pdf file
        :param layout: whether to retain the original physical layout for a page. If disabled, PDF pages are read in
                       the content stream order.
        """
        if layout:
            command = ["pdftotext", "-enc", "Latin1", "-layout", str(file_path), "-"]
        else:
            command = ["pdftotext", "-enc", "Latin1", str(file_path), "-"]
        output = subprocess.run(command, stdout=subprocess.PIPE, shell=False)  # type: ignore
        document = output.stdout.decode(errors="ignore")
        pages = document.split("\f")
        pages = pages[:-1]  # the last page in the split is always empty.
        return pages

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

    def clean(self,
              document,
              clean_whitespace=True,
              remove_header_and_footer=True,
              clean_empty_lines=True):
        """
        Perform document cleaning on a single document and return a single document. This method will deal with whitespaces, headers, footers
        and empty lines -- roughly based on haystack.
        """
        lines = document.splitlines()

        if remove_header_and_footer:
            # simplest way fo removing header and footer
            lines = lines[1:-2]

        if clean_whitespace:
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                cleaned_lines.append(line)
            text = " ".join(cleaned_lines)

        if clean_empty_lines:
            text = re.sub(r"\n\n+", "\n\n", text)
            text = re.sub(r"[\s]+", " ", text)

        return text

    def process_list_of_text(self, list_of_texts):
        processed_list = []
        for sent in list_of_texts:
            tokens, indices = self.tokenizer_wrapper(sent, self.tokenizer)
            processed_list = [str(vocab_idx) for vocab_idx in indices]

        if processed_list:
            return processed_list

    def pdffile_to_list_of_sentences(self, input_file):
        """
        Prepare an input txt file to a list of sentences (following the settings of using subwordunits, stemming,
        stopwords), so it can be added to a single corpus to compute the IDF weights.

        :param input_file: json format containing results from converting pdftotext
        :return: list of processed sentences
        """
        pages = self.read_pdf(input_file, layout=True)  # has to be set to true
        processed_list_of_sentences = []
        for page in pages:
            text = self.clean(page)
            all_sentences = []
            for part in text.split('\n'):
                all_sentences += [str(s) for s in TextBlob(part).sentences]

            processed_sent = self.process_list_of_text(all_sentences)
            if processed_sent:
                processed_list_of_sentences += processed_sent

        return processed_list_of_sentences

    def compute_IDF_weights(self, overwrite=True):
        """
        Overarching function to compute or load the IDF weights, as well as train or load a SentencePiece model - based
        on the settings provided to :class:`~SORE.my_utils.PrepIDFWeights`

        """
        # computed weights
        if os.path.exists(self.IDF_path) and not overwrite:
            print("Loading existing IDF weights.")       # this part is deprecated
            with open(self.IDF_path, 'r') as f:
                IDF = json.load(f)
        else:
            input_files = glob.glob(self.input_file_dir + '**/*.pdf', recursive=True)
            print("[IDF IDF IDF] ", self.input_file_dir)
            corpus_list = []
            if len(input_files) < 1:
                print(
                    "No input files found! Make sure you pass an input file directory with txt files containing lines with sentence(s)")
                return 0
            else:
                total = len(input_files)
                print(f"Combining sentences from {total} pdf files into a single corpus to compute IDF weights.")
                for input_file in tqdm(input_files):
                    corpus_list.append(self.pdffile_to_list_of_sentences(input_file))

            # Could add additional terms in here, e.g., the terms from the vocabularies etc.
            # if additional_text:
            #     corpus_list += self.process_list_of_text(additional_text)

            IDF = self.get_idf(corpus_list)

            with open(self.IDF_path, 'w') as f:
                json.dump(IDF, f)

        # print some tokens and IDF values to see what kind of stuff we get
        print("Printing some IDF values, should be subword units!")
        sanity_check = [x for x in IDF.keys()]
        for x in sanity_check[:10]:
            print(self.detokenizer_wrapper(x, self.tokenizer))

        return self.IDF_path



