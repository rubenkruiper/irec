from typing import List, Optional, Dict, Any, Generator, Set
import json
import re
import copy

from functools import partial, reduce
from itertools import chain

from tqdm import tqdm
from transformers import BertTokenizer
# from tokenizers import BertWordPieceTokenizer


def custom_cleaning_rules(objects):
    """
    objects can be a List[str] or str
    """
    input_type = 'list'
    if type(objects) == str:
        input_type = 'str'
        objects = [objects]

    cleaned_objects = []
    for obj in objects:
        # remove double determiners that are sometimes grabbed, and strip objects
        obj = obj.replace("thethe", '', 1).strip()
        obj = obj.replace("thenthe", '', 1).strip()
        obj = obj.replace("thethat", '', 1).strip()
        obj = obj.replace("their ", '', 1).strip()
        obj = obj.replace(". ", '').strip()

        if len(obj) == 1:
            # remove 1 character objects
            continue
        elif len(obj) < 4 and (not obj.isupper() or any(c for c in obj if (c.isdigit() or c.isspace()))):
            # remove 2 & 3 characters objects that aren't all uppercase (abbreviations?) / contain a number or space
            # while removing some 3 letter words like 'ice' and 'fan', most of these are uninformative/erroneous
            continue
        elif len(obj) < 6 and len(re.findall(r"[^\w\s]", obj)) > 1:
            # any span of 5 characters or less, that contains multiple non-word and non-space characters
            continue
        elif len(re.findall(r"[=+*@|<>»_%]", obj)) > 0:
            # any span that may indicate that its taken from an equation or email address or simply gibberish from ocr
            continue
        elif obj.startswith("the ") or obj.startswith("The ") or obj.startswith("a ") or obj.startswith("A "):
            # do the same 1 char and 2/3 char removal in case the object starts with a determiner;
            if len(obj) == 5:
                continue
            elif len(obj) < 8 and obj[4:].islower():
                continue
            else:
                cleaned_objects.append(obj)
        else:
            cleaned_objects.append(obj)
    if input_type == 'list':
        return list(set(cleaned_objects))
    if input_type == 'str':
        try:
            return cleaned_objects[0]
        except IndexError:
            return ''


class RegexFilter:
    def __init__(self):
        pass

    def single_regex_pattern(self, some_pattern, texts):
        """
        Helper function that applies a single regex pattern to a list of textspans.
        """
        no_updated = 0
        p = re.compile(some_pattern)
        new_texts = texts
        removed = []
        for idx, t in enumerate(texts):
            # check that the object wasn't already removed in a previous pass
            if t != '':
                t_ = re.sub(p, '', t)
                new_texts[idx] = t_

                if not t_:
                    # empty str after re.sub
                    removed.append(t)
                elif t_ != t:
                    # different str after re.sub
                    no_updated += 1

        # print("Removed {} objects, updated {}".format(len(removed), no_updated))
        return new_texts, removed

    def run_filter(self, to_be_filtered, regex_dict=None):
        """
        Function that can be called to run a set of regular expressions to filter out specific spans or parts of spans.
        Basic regexes are provided for identifying title_numbers, references and gibberish numbers in text.
        """
        if not regex_dict:
            # todo: improve regexes for preprocess-filtering
            regex_dict = {
                'title_numbers': '^([A-Z]{1}[. ]{1})?([ \d.+-])*',      # (?<![: ])(?![\D\-:]{1})
                'references': '[(]+([\d\s.])*[)]?',
                'gibberish_numbers': '^(\d|\w|_|—|@|=|\/|\\|~|\.|,|<|>|:|°|\*|\||\(|\))(?(1)(\s?(\d|_|—|@|=|\/|\\|~|\.|,|<|>|:|%|\*|\||\(|\))\s?)+(\w(?!\w))?|)',
                #     'real_numbers': '^\d*((\s)?(.|,)?(\s)?\d)*$',
            }

        removed_objects = []
        if type(to_be_filtered) == str:
            to_be_filtered = [to_be_filtered]

        updated_objects = copy.deepcopy(to_be_filtered)

        for filter_type, pattern in regex_dict.items():
            # print("Filtering {}".format(filter_type))
            updated_objects, removed = self.single_regex_pattern(pattern, updated_objects)
            removed_objects += removed

        if '' in updated_objects:
            updated_objects.remove('')

        return removed_objects, updated_objects

def remove_unicode_chars(text):
    text = text.replace("\u00a0", ' ')              # no-break-space
    text = text.replace("\u00a3", 'pounds ')        # £
    text = text.replace("\u00b2", '#SUP#2#SUP#')    # superscript 2
    text = text.replace("\u00b3", '#SUP#3#SUP#')    # superscript 2
    text = text.replace("\u00b0", ' degrees ')      # degrees sign
    text = text.replace("\u00ba", ' degrees ')      # degrees sign ~ masculine ordinal coordinator
    text = text.replace("\u00bd", '1/2')            # vulgar fraction half
    text = text.replace("\u00be", '3/4')            # vulgar fraction quarter
    text = text.replace("\u03bb", 'lambda')         # λ lambda
    text = text.replace("\u00e9", 'e')              # é
    text = text.replace("\u2013", '-')              # en-dash
    text = text.replace("\u2014", '-')              # em-dash
    text = text.replace("\xe2", '-')                # dash
    text = text.replace("\u2018", '`')              # left-single quotation mark
    text = text.replace("\u201c", '``')             # left-double quotation mark
    text = text.replace("\u2019", "'")              # right-single quotation mark
    text = text.replace("\u201d", "''")             # right-double quotation mark
    text = text.replace("\u2026", "...")            # horizontal ellipses
    text = text.replace("\uf059", "PSI")            # psi sign
    text = text.replace("\u00f7", "/")              # psi sign
    text = text.replace("\u2028", '\n')             # line separator
    text = text.replace("\xa0", " ")                # space
    text = text.replace("\xe2\x96\xba", "")         # arrow right
    text = text.replace("\xe2\x97\x84", "")         # arrow left
    return text


### Haystack functions that I'm reusing in the notebook
def find_and_remove_header_footer(pages: List[str], n_chars: int, n_first_pages_to_ignore: int, n_last_pages_to_ignore: int
) -> str:
    """
    Heuristic to find footers and headers across different pages by searching for the longest common string.
    For headers we only search in the first n_chars characters (for footer: last n_chars).
    Note: This heuristic uses exact matches and therefore works well for footers like "Copyright 2019 by XXX",
     but won't detect "Page 3 of 4" or similar.

    :param n_chars: number of first/last characters where the header/footer shall be searched in
    :param n_first_pages_to_ignore: number of first pages to ignore (e.g. TOCs often don't contain footer/header)
    :param n_last_pages_to_ignore: number of last pages to ignore
    :return: (cleaned pages, found_header_str, found_footer_str)
    """

    # header
    start_of_pages = [p[:n_chars] for p in pages[n_first_pages_to_ignore:-n_last_pages_to_ignore]]
    found_header = find_longest_common_ngram(start_of_pages)
    if found_header:
        pages = [page.replace(found_header, "") for page in pages]

    # footer
    end_of_pages = [p[-n_chars:] for p in pages[n_first_pages_to_ignore:-n_last_pages_to_ignore]]
    found_footer = find_longest_common_ngram(end_of_pages)
    if found_footer:
        pages = [page.replace(found_footer, "") for page in pages]

    return pages

def ngram(seq: str, n: int) -> Generator[str, None, None]:
    """
    Return ngram (of tokens - currently split by whitespace)
    :param seq: str, string from which the ngram shall be created
    :param n: int, n of ngram
    :return: str, ngram as string
    """

    # In order to maintain the original whitespace, but still consider \n and \t for n-gram tokenization,
    # we add a space here and remove it after creation of the ngrams again (see below)
    seq = seq.replace("\n", " \n")
    seq = seq.replace("\t", " \t")

    words = seq.split(" ")
    ngrams = (
        " ".join(words[i: i + n]).replace(" \n", "\n").replace(" \t", "\t") for i in range(0, len(words) - n + 1)
    )

    return ngrams

def allngram(seq: str, min_ngram: int, max_ngram: int) -> Set[str]:
    lengths = range(min_ngram, max_ngram) if max_ngram else range(min_ngram, len(seq))
    ngrams = map(partial(ngram, seq), lengths)
    res = set(chain.from_iterable(ngrams))
    return res

def find_longest_common_ngram(sequences: List[str], max_ngram: int = 30, min_ngram: int = 3) -> Optional[str]:
    """
    Find the longest common ngram across different text sequences (e.g. start of pages).
    Considering all ngrams between the specified range. Helpful for finding footers, headers etc.

    :param sequences: list[str], list of strings that shall be searched for common n_grams
    :param max_ngram: int, maximum length of ngram to consider
    :param min_ngram: minimum length of ngram to consider
    :return: str, common string of all sections
    """
    sequences = [s for s in sequences if s]  # filter empty sequences
    if not sequences:
        return None
    seqs_ngrams = map(partial(allngram, min_ngram=min_ngram, max_ngram=max_ngram), sequences)
    intersection = reduce(set.intersection, seqs_ngrams)

    try:
        longest = max(intersection, key=len)
    except ValueError:
        # no common sequence found
        longest = ""
    return longest if longest.strip() else None