from typing import List, Union, Dict, Any

import torch
import pickle
import os.path
import numpy as np

from pathlib import Path
from transformers import BertModel, BertTokenizer


class Embedder:
    def __init__(self, tokenizer: BertTokenizer,
                 bert: BertModel,
                 IDF_dict: Dict[str, float],
                 embedding_fp: Path,
                 layers_to_use: List[int] = [12],
                 layer_combination: str = "avg",
                 idf_threshold: float = 1.5,
                 idf_weight_factor: float = 1.0,
                 not_found_idf_value: float = 0.5):
        """
        Object that provides functinoality to tokenize some input text, as well as computes the
        (potentially IDF weighted) embeddings.

        :param tokenizer:    Pretrained BertTokenizer.
        :param bert:    Pretrained BERT model - from the same BERT model as the tokenizer.
        :param IDF_dict:    Dictionary holding the pre-computed IDF weights for token indices.
        :param layers_to_use:   A list with layer indices to combine. Default is [12], meaning only the output
                                of the last layer is used.
        :param layer_combination:   In case of multiple layers, determine how to combine them. Defaults to `"avg"`,
                                    choice between (`"avg"`, `"sum"`, `"max"`).
        :param idf_threshold:    Minimum IDF weight value for the token to be included during weighting.
        :param idf_weight_factor:    Factor for multiplying IDF weights to stress/reduce the difference between high/low.
        :param not_found_idf_value:    Value for tokens that do not have a corresponding IDF weight(typically low).
        """
        self.tokenizer = tokenizer
        self.bert = bert
        self.IDF_dict = IDF_dict
        self.layers_to_use = layers_to_use
        self.layer_combination = layer_combination
        if idf_threshold <= 0  or not_found_idf_value  <= 0 or idf_weight_factor <= 0:
            print("The idf_threshold, idf_weight_factor and not_found_idf_value should be bigger than 0! Setting to defaults")
            self.idf_threshold = 1.5
            self.idf_weight_factor = 1.0
            self.not_found_idf_value = 0.5
        else:
            self.idf_threshold = idf_threshold
            self.idf_weight_factor = idf_weight_factor
            self.not_found_idf_value = not_found_idf_value
        
        
        self.embedding_fp = embedding_fp
        self.emb_mean_fp = embedding_fp.joinpath("standardisation_mean.pkl")
        self.emb_std_fp = embedding_fp.joinpath("standardisation_std.pkl")
        self.emb_mean = "None"  # initialise to specific value, later will hold an array (without truth value blabla)
        self.emb_std = "None"

    def prepare_tokens(self, text: str) -> (List[str], List[int]):
        """
        Helper function to tokenize and ensure the same output is provided by the BertTokenizer
        (and BertWordPieceTokenizer ~ deprecated).

        :param tokenizer:    A pretrained BertTokenizer.
        :param text:    Input text.
        :return tokenized_text: List of string representations of tokens for the input text.
        :return indices:    List of token index representations of tokens for the input text.
        """
        # if type(self.tokenizer) == BertWordPieceTokenizer:
        #     encoded = self.tokenizer.encode(text)
        #     tokenized_text = encoded.tokens
        #     indices = encoded.ids
        #     return tokenized_text, indices
        if type(self.tokenizer) == BertTokenizer:
            tokenized_text = self.tokenizer.tokenize("[CLS] " + text + " [SEP]")
            indices = self.tokenizer.convert_tokens_to_ids(tokenized_text)

            if len(tokenized_text) > 512:
                # Any tokens where the index is higher than 512 (max BERT input) will be omitted
                # todo; make this adjustable for different pretrained embedding models like a longformer model
                print("Number of input tokens too long! Truncating input to 512 tokens...")
                tokenized_text, indices = tokenized_text[:512], indices[:512]

            return tokenized_text, indices
        else:
            print("You'll need to change to a BertTokenizer")
            return None  # need to raise appropriate error or quit running nicely

    def get_IDF_weights_for_indices(self,
                                    tokenized_text: List[str],
                                    indices: List[int]) -> np.array:
        """
        Helper function to get the relevant IDF weights given tokenized input. Note: if an IDF value is not found, then
        the weight for that token is set to `self.not_found_idf_value`.

        :param tokenized_text:  List of tokens, currently not used. Previously used to visualise which part of the
                                input text is filtered through IDF-based filtering, and which weights are assigned.
        :param indices: Token-indices that are also used to index into the corresponding IDF values, used to retrieve
                        the corresponding IDF weights computed for a domain corpus.
        :return sw_weights: An np.array of the same length as the parameter `indices`, holding IDF weights for the
                            subword unit tokens.
        """
        sw_weights = np.ones(len(indices))
        visualised_text = ''
        for sw_id, sw in enumerate(indices):
            try:
                # set index of phrase representation to corresponding IDF value
                sw_weights[sw_id] = self.IDF_dict[str(sw)]
                visualised_text += ' ' + tokenized_text[sw_id]
            except (KeyError, ValueError) as e:

                # No IDF value found, which value do we set it to?
                sw_weights[sw_id] = self.not_found_idf_value
                visualised_text += ' ' + tokenized_text[sw_id] + '\u0336'

        return sw_weights

    def embed_text(self, text: str) -> List[torch.tensor]:
        """
        :param text:    Text to be tokenized. Expecting BERT-tokenizer, so max input of 512 tokens.

        :return weighted_t_embs:    The embeddings for the each of the tokens in the sentence, potentially with  subword
                                    IDF weights applied (multiplied by IDF value, mediated by `self.idf_weight_factor`).
        """
        tokenized_text, indices = self.prepare_tokens(text)

        IDF_weights = self.get_IDF_weights_for_indices(tokenized_text, indices)

        segments_ids = [1] * len(indices)
        tokens_tensor = torch.tensor([indices])
        segments_tensors = torch.tensor([segments_ids])
        embedding = self.bert(tokens_tensor, segments_tensors)

        try:
            hidden_states = embedding[2]
        except IndexError:
            print("Make sure to output hidden states; BertModel.from_pretrained(some-bert-model, "
                  "output_hidden_states=True)")

        # Group by token vectors
        token_embeddings = torch.stack(hidden_states, dim=0).squeeze(dim=1).permute(1, 0, 2)
        weighted_t_embs = []
        for token, IDF_w in zip(token_embeddings, IDF_weights):
            # todo -- consider running more experiments with different BERT layers and combinations
            # Combine the vectors from selected bert layers --> currently default to last layer only [12]
            if self.layer_combination == "sum":
                combined_vec = torch.sum(token.index_select(0, torch.tensor(self.layers_to_use)), dim=0)
            elif self.layer_combination == "avg":
                combined_vec = torch.mean(token.index_select(0, torch.tensor(self.layers_to_use)), dim=0)
            elif self.layer_combination == "max":
                combined_vec = torch.max(token.index_select(0, torch.tensor(self.layers_to_use)), dim=0)

            # Weight the vector by IDF value
            if IDF_w > self.idf_threshold and self.idf_weight_factor > 0:
                # avoid multiplying by 0
                weighted_t_embs.append(combined_vec * (self.idf_weight_factor * IDF_w))
            else:
                # the IDF weight does not meet the threshold
                weighted_t_embs.append(combined_vec * 0.001)

        return weighted_t_embs
    
    def embed_a_span(self, span: str) -> torch.tensor:
        """
        Use to simply embed a span, BEFORE the mean and std of all spans are computed
        """
        embeddings = self.embed_text(span)
        try:
            return (span, torch.stack(embeddings).detach().numpy().squeeze())
        except RuntimeError:
            # can happen if the tensor for the span is empty somehow
            print(f"Empty tensor! Not sure why, but will drop the span: {span}")
            return None
    
    def embed_and_normalise_span(self, span: str) -> torch.tensor:
        """
        Use to embed new spans, AFTER the mean and std of all spans are computed.
        Returns a tuple: (span, normalised_embedding)
        """
        embeddings = self.embed_text(span)
        try:
            return (span, self.combine_token_embeddings(embeddings))
        except RuntimeError:
            # can happen if the tensor for the span is empty somehow
            print(f"Empty tensor! Not sure why, but will drop the span: {span}")
            return None
    
    def embed_and_normalise(self, span: str) -> torch.tensor:
        """
        Use to embed new spans, AFTER the mean and std of all spans are computed
        Returns a normalised embedding.
        """
        embeddings = self.embed_text(span)
        try:
            return self.combine_token_embeddings(embeddings)
        except RuntimeError:
            # can happen if the tensor for the span is empty somehow
            print(f"Empty tensor! Not sure why, but will drop the span: {span}")
            return None
    

    def combine_token_embeddings(self, embeddings: List[torch.tensor]) -> torch.tensor:
        """
        Calls the embedding function for a span, stacking the embeddings for each token and computing the mean over
        the token-length dimension.

        :param embeddings:   List of tensors for each of the tokens in a given span.
        :return embedding:   Single embedding of the span, as an average over the constituent token embeddings.
        """
        weighted_token_embeddings = torch.stack(embeddings)
        detached_embeddings = weighted_token_embeddings.detach().numpy()
        # Standardise PLM representation
        if self.emb_mean == "None" and self.emb_mean_fp.exists():
            self.emb_mean = pickle.load(open(self.emb_mean_fp, 'rb'))
            self.emb_std = pickle.load(open(self.emb_std_fp, 'rb'))

        standardised_embeddings = (detached_embeddings - self.emb_mean) / self.emb_std
        return np.mean(standardised_embeddings, axis=0)   # average over the tokens for now. old >> torch.mean(x, dim=0)

    def embed_and_stack(self, span: str) -> torch.tensor:
        """
        Combination of above function to simplify calls from the a threadpool executor. Returns tuple of span and its
        corresponding embedding.
        """
        embeddings = self.embed_text(span)
        return (span, self.combine_token_embeddings(embeddings))
