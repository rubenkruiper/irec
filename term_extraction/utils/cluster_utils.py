from typing import Dict, List, Any
import os, glob, pickle, json, concurrent, math
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from Levenshtein import distance as lev
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine, euclidean

from utils.embedding_utils import Embedder
import utils.cleaning_utils as cleaning_utils

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# TODO; break this up into separate files :)
    
############################################################################################################
###### Elbow score and silhouette score to help determine a reasonable value for K
class ElbowAndSilhouette:
    def __init__(self, cluster_dir: Path, elbow:bool=True, silhouette:bool=True):
        self.sum_of_squared_distances = []
        self.silhouette_avg_euclidean = []
        self.silhouette_avg_cosine = []
        self.elbow = elbow
        self.silhouette = silhouette
        
        self.cluster_dir = cluster_dir
        self.cluster_spans = pickle.load(open(cluster_dir.joinpath("unique_spans.pkl"), 'rb'))
        self.cluster_data = pickle.load(open(cluster_dir.joinpath("standardised_embeddings.pkl"), 'rb'))

    def compute_sum_of_squared_distances(self, centroids_, assignments_):
        temp_index = pd.MultiIndex.from_arrays([assignments_, self.cluster_spans], 
                                                names=('assigned_cluster','text'))

        squared_distance_per_span = pd.Series([np.sum(np.absolute(emb - centroids_[assignments_[idx]]))**2 for idx, emb in enumerate(self.cluster_data)], 
                              index=temp_index, 
                              name="distance_to_centroid")

        all_squared_distances_for_label = [squared_distance_per_span.get(x) for x in range(len(set(assignments_)))]

        sum_of_squared_values = 0
        for idx, squared_distances in enumerate(all_squared_distances_for_label):
            try:
                sum_of_squared_values += np.sum(squared_distances)
            except:
                print("Dead cluster ID: ", idx, '  - ', squared_distances)
            
        return sum_of_squared_values

    def compute_scores_for_single_model(self, pkl_name: Path):
        centroids, assignments = pickle.load(open(pkl_name, 'rb'))

        # Drop NAN clusters that 'died' if they exist
        mask = ~np.isnan(centroids).any(axis=1)
        for idx, m in enumerate(mask):
            if not m:
                # placeholder so we don't have to deal with shifting indices of assigned clusters (out-of-index)
                centroids[idx] = np.array([0.0001]*768)

        num_clusters = len(centroids)

        if self.silhouette and num_clusters > 12000:
            print("""
                For large numbers of clusters (e.g. > 10K), computing the silhouette score can run into memory
                errors. Therefore, I'll turn this off.
                """)
            silhouette = False
        else:
            silhouette = self.silhouette

        if self.elbow:
            self.sum_of_squared_distances.append(self.compute_sum_of_squared_distances(centroids, assignments))

        if silhouette:
            self.silhouette_avg_euclidean.append(silhouette_score(self.cluster_data, 
                                                             assignments, metric="euclidean"))
            self.silhouette_avg_cosine.append(silhouette_score(self.cluster_data, 
                                                               assignments, metric="cosine"))


    def save_and_plot_with_scores(self, clustering_type:str, pkl_files:List[Path]):
        print("Plotting the figure")
        # ensure order is based on the number of clusters
        dict_with_values = {}
        for idx, x in enumerate(pkl_files):
            try:
                num_clusters = int(x.stem.split('_')[1])
                dict_with_values[num_clusters] = {
                                           "num_clusters": num_clusters,
                                           "sum_of_squared_distances": self.sum_of_squared_distances[idx],
                                           "silhouette_avg_euclidean": self.silhouette_avg_euclidean[idx],
                                           "silhouette_avg_cosine": self.silhouette_avg_cosine[idx]
                }
            except:
                continue

        dict_to_plot = {
               "num_clusters": [],
               "sum_of_squared_distances": [],
               "silhouette_avg_euclidean": [],
               "silhouette_avg_cosine": []
            }
        ordered_keys = [k for k in dict_with_values.keys()]
        ordered_keys.sort()
        for k in ordered_keys:
            for v in dict_with_values[k]:
                dict_to_plot[v].append(dict_with_values[k][v])

        # create a pandas DataFrame that we'll save and plot
        df_to_plot = pd.DataFrame(dict_to_plot)
        # save 
        name_for_file = self.cluster_dir.joinpath(clustering_type + "_" + \
                        "_".join([str(n) for n in dict_to_plot['num_clusters']]) + "_.csv")
        df_to_plot.to_csv(name_for_file)
        # plot
        sn.set_style("darkgrid", {"axes.facecolor": ".95"})
        
        fig, ax1 = plt.subplots(figsize=(len(dict_to_plot['num_clusters'])/2, 4), dpi=120)  
        
        ax1.xaxis.set_ticks(df_to_plot['num_clusters'], 
                            labels=[str(int(x)/1000)+"K" for x in df_to_plot['num_clusters']],
                           fontsize=9)
        
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        
        colour1 = 'seagreen'
        colour2 = 'salmon'
        colour3 = 'purple'
        ax1.tick_params(axis = 'y', colors=colour1)
        ax2.tick_params(axis = 'y', colors=colour2)
        ax3.tick_params(axis = 'y', colors=colour3)
        
        ax1.plot(df_to_plot['num_clusters'], 
                 df_to_plot["sum_of_squared_distances"], 
                 color=colour1, label='Sum of Squared Errors')
        ax2.plot(df_to_plot['num_clusters'], 
                 df_to_plot["silhouette_avg_euclidean"], 
                 color=colour2, label='Silhouette Euclidean')
        ax3.plot(df_to_plot['num_clusters'], 
                 df_to_plot["silhouette_avg_cosine"], 
                 color=colour3, label='Silhouette Cosine')
        
        ax1.legend().remove()
        plt.gcf().legend(bbox_to_anchor=(.9, 0.6))
        
        
#         df_to_plot.set_index('num_clusters').plot(kind='line', 
#                                                   color=['seagreen', 'salmon', 'red'],
#                                                   fontsize=12,
#                                                   subplots=True)


    def compute_scores_for_models(self, clustering_type:str, pkl_files:List[Path]):

        print("Computing elbow and silhouette (if not too many num_clusters) scores.")
        csv_files = self.cluster_dir.glob(clustering_type + "*_.csv")
        if csv_files:
            csv_file = max(csv_files) # bit hacky, the longest file has the most cluster info stored
            latest_df = pd.read_csv(csv_file)
        else:
            csv_file = self.cluster_dir.joinpath("not.created")
            
        updated_pkl_files = []
        for pkl_name in tqdm(pkl_files):
            if clustering_type not in pkl_name.name:
                print(f"{pkl_name} is not of expected clustering type? {clustering_type}")
                continue

            num_clusters = pkl_name.name.split('_')[1]
            try:
                int(num_clusters)
            except:
                print(f"File may not have the right naming format: {pkl_name}")
                continue

            updated_pkl_files.append(pkl_name)
       
            if num_clusters in csv_file.stem.split('_'):
                print(f"Loading values from existing csv file: {pkl_name}")
                # already computed so we'll reuse the values
                temp = latest_df.loc[latest_df['num_clusters'] == int(num_clusters)]
                self.sum_of_squared_distances.append(temp['sum_of_squared_distances'].tolist()[0])
                self.silhouette_avg_euclidean.append(temp['silhouette_avg_euclidean'].tolist()[0])
                self.silhouette_avg_cosine.append(temp['silhouette_avg_cosine'].tolist()[0])
            else:
                # compute value if needed
                print(f"Working on: {pkl_name}")
                self.compute_scores_for_single_model(pkl_name)

        self.save_and_plot_with_scores(clustering_type, updated_pkl_files)


    
############################################################################################################
###### Dictionary to store clusters of spans
class ClusterDict:
    def __init__(self, 
                 ref_corp_unique: List[str],
                 unique_spans: List[str], 
                 unique_clustering_data, 
                 centroids, 
                 assignments,
                 embedding_fp: Path = Path("output/")
                 ):
        self.ref_corp_unique = ref_corp_unique
        self.unique_spans = unique_spans
        self.unique_clustering_data = unique_clustering_data
        self.centroids = centroids
        self.assignments = assignments
        self.embedding_fp = embedding_fp

    def single_cpu_task(self, idx):
        """
        Single CPU task to update span_dict, for parallel execution using `concurrent.futures`. Computes the distance
        between (1) the embedding of some span and (2) the vector that represent the centroid of the cluster assigned
        to that span.

        :param idx: The index into 1) `self.assignments`, 2) `self.unique_clustering_data` and 3) `self.centroids`.
                    These respectively represent the assigned cluster ID, the embedding vector, and a vector for
                    cluster centroid.
        :return dict:   A dictionary of format `{span_idx: {'span': unique_spans[idx],
                                                                'label': int(label),
                                                                'distance': float(distance)}`
        """
        label = self.assignments[idx]
        distance = np.sum(np.absolute(self.unique_clustering_data[idx] - self.centroids[label]))
        return {idx: {'span': self.unique_spans[idx],
                      'label': int(label),
                      'distance': float(distance)}}

    def compute_span_dict(self, unique_spans: List[str], max_num_cpu_threads: int = 8) -> Dict[str, Any]:
        """
        Compute a dictionary holding distances to centroids, filtered by centroid labels. Uses `concurrent.futures`
        to speed up the processing, with default number of cpu threads.

        :param unique_spans:    List of unique spans (no duplicates expected) for which to determine the centroid label,
                                and distance to that centroid.
        :return span_dict:  A dictionary of format `{span_idx: {'span': unique_spans[idx],
                                                                'label': int(label),
                                                                'distance': float(distance)}`
        """
        span_dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_num_cpu_threads) as executor:
            futures = [executor.submit(self.single_cpu_task, idx) for idx in range(len(unique_spans))]

        returned_dicts = [f.result() for f in futures]
        [span_dict.update(r) for r in returned_dicts]
        return span_dict

    def convert_to_cluster_dict(self, span_dict):
        """
        Reorder the span_dict dictionary `{span_id: {label: ..., distance: ..., span: ...}` to a dict where keys are
        the cluster labels; `{cluster_id: {span (text): ..., distance(to centroid): ...}}`.

        :param span_dict:   A dictionary that holds clustering information accessible by unique span_id.
        :return cluster_dict:   A dictionary that holds clustering information accessible by unique cluster_id.
        """
        # reorder to dictionary with cluster_ids as
        cluster_dict = {}
        for span in span_dict.values():
            if str(span['label']) not in cluster_dict.keys():
                cluster_dict[str(span['label'])] = [[span['distance'], span['span']]]
            else:
                cluster_dict[str(span['label'])].append([span['distance'], span['span']])

        # reorder list of terms per cluster, based on distance
        for cluster_id in cluster_dict.keys():
            cluster_dict[str(cluster_id)] = sorted(cluster_dict[cluster_id], key=lambda s: (s[0]))
        return cluster_dict

    def prep_cluster_dict(self, chosen_num_clusters:int):
        """
        Loads or prepares the dictionary that holds information on clusters; spans and their respective distances
        to the centroid of a cluster. A file called `phrase_cluster_dict.json`, always stored at `self.embedding_fp`.

        :return phrase_cluster_dict:   Dictionary holding `{cluster ID: [[distance, span], ...]}`.
        """
        cluster_dict_filepath = self.embedding_fp.joinpath(f"cluster_dict_{chosen_num_clusters}.json")
        clusters_to_filter_filepath = cluster_dict_filepath.parent / f"clusters_to_filter_{chosen_num_clusters}.pkl"
        if cluster_dict_filepath.exists():
            print("Loading a pre-computed cluster dictionary from file.")
            phrase_cluster_dict = json.load(open(cluster_dict_filepath, 'r'))
            clusters_to_filter = set(pickle.load(open(clusters_to_filter_filepath, 'rb')))
        else:
            print("Computing the cluster dictionary.")
            span_dict = self.compute_span_dict(self.unique_spans)
            phrase_cluster_dict = self.convert_to_cluster_dict(span_dict)
            with open(cluster_dict_filepath, 'w') as f:
                json.dump(phrase_cluster_dict, f)

            unwanted_terms = set(self.ref_corp_unique)
            clusters_to_filter = {}
            for c_id, dists_and_terms in phrase_cluster_dict.items():
                term_list = [t for _, t in dists_and_terms]
                unwanted_terms_in_cluster = [t for t in term_list if t in unwanted_terms]
                # clusters that contain background terms are stored 
                if unwanted_terms_in_cluster:
                    if c_id not in clusters_to_filter:
                        clusters_to_filter[c_id] = unwanted_terms_in_cluster

            # store the clusters_to_keep for filtering later on
            with open(clusters_to_filter_filepath, 'wb') as f:
                pickle.dump(clusters_to_filter, f)

        return phrase_cluster_dict, clusters_to_filter
    
    
############################################################################################################
###### cluster_data container

def levenshtein(w1, w2):
    """
    Determine the Levenshtein distance between two spans, divided by the length of the longest span. If this
    value is below a given threshold (currently hardcoded to .75) the spans are considered dissimilar. This
    function is used to retrieve 'similar spans' from a cluster, with the aim to return spans that aren't close
    too similar to the given span; e.g., if the span is `test` the aim is to not return `tests` or `Test`.

    :param w1:  First of two spans to compare.
    :param w2:  Second of two spans to compare.
    :return bool:   Returns True for words that share a lot of characters, mediated by the length of the
                    longest input.
     """
    #   More character overlap --> smaller levenshtein distance
    # remove determiner
    if w1.startswith('the ') or w1.startswith('a ') or w1.startswith('The ') or w1.startswith('A '):
        w1 = w1.split(' ', 1)[1]
    if w2.startswith('the ') or w2.startswith('a ') or w2.startswith('The ') or w2.startswith('A '):
        w2 = w2.split(' ', 1)[1]

    if len(w1) > len(w2):
        long_w, short_w = w1, w2
    else:
        short_w, long_w = w1, w2

    # Comparison is between lowercased words, in order to ignore case
    # % 75% character level similarity minimum
    return 100 - (lev(short_w.lower(), long_w.lower()) / len(long_w) * 100) > 75


class ToBeClustered:
    def __init__(self, text: str, embedder: Embedder):
        """
        Object that holds a text-span and provides access to embedding-related cluster_data; tokenized_text,
        token_indices, IDF_values_for_tokens, embedding.
        """
        self.text = text

        self.tokenized_text, self.token_indices = embedder.prepare_tokens(text)
        self.IDF_values_for_tokens = embedder.get_IDF_weights_for_indices(self.tokenized_text, self.token_indices)
        self.embedding = embedder.combine_token_embeddings(embedder.embed_text(text))

        # placeholders for cluster_ID and neighbours
        self.cluster_id = -1
        self.distance_to_centroid = math.inf
        self.all_neighbours = []

        self.idf_threshold = embedder.idf_threshold

    def print_tokens_and_weights(self, idf_threshold: float = None):
        """ Aim is to check how the IDF threshold affects the influence of subword tokens on the entire span weight. """
        if not idf_threshold:
            idf_threshold = self.idf_threshold

        subword_insight(self.tokenized_text, self.IDF_values_for_tokens, self.text, idf_threshold)

    def get_top_k_neighbours(self, unique_span_dict: Dict[str, Any], cosine_sim_threshold: float = 0.7, top_k: int = 3):
        """
        Function to compute the `top_k` terms in a cluster that are closest to it's centroid. The idea is to compute
        this list for the spans in a passage to-be-retrieved (before indexing) and for a query (during query-expansion).
        The aim is thus to increase the similarity between query and document at the level of related terms.
        Cosine similarity is used to further ensure embedding similarity, and levenshtein distance is used to make sure
        the neighbours aren't simply the plural form.

        :param unique_span_dict:   Dict with unique spans as keys and their embeddings as values, original input to
                                   the KMeans clustering.
        :param cosine_sim_threshold:   Minimum cosine similarity value for a neighbour to be considered 'similar'.
        :param top_k:   An `int` value to determine the maximum number of related strings to return.
        :return all_top_terms:  A `list` of terms from the assigned cluster.
        """
        # List that will hold the neighbours. Each new candidate will be compared to the spans in this list, so we
        # include the original span itself as well for that comparison and later omit it.
        all_top_terms = [self.text]
        inflections  = [self.text]
        # sort neighbours (takes a long time but seems to work well... maybe I should try hierarchically cluster or smt)
        #  - euclidean distance for now;
        self.all_neighbours.sort(key=lambda x: euclidean(unique_span_dict[x[1]], self.embedding))

        for dist, neighbour in self.all_neighbours:
            # first run the custom cleaning rules on the potential neighbours; 
            #  Not really necessary, but if you update the cleaning rules without clustering again,
            #  then this is where it reflects in the results.
            neighbour = cleaning_utils.custom_cleaning_rules(neighbour)
            if not neighbour:
                continue

            # If the Levenshtein distance to other already added neighbours is too close (True), skip this neighbour
            if any([levenshtein(already_added, neighbour) for already_added in all_top_terms]):
#                 print(f"[possible inflection] {self.text} ~ {neighbour}")
                inflections.append(neighbour)
                continue

            # Get the embedding for this neighbour span
            # todo; very often a neighbour doesn't exist in the unique_span dict, but these are usually
            #  the weirdest spans, like "Tel +44..."
            #  - they come from the phrase_cluster_dict, which are all spans that have been clustered
            #  - they don't exist in unique_span_dict, because... they may be part of clusters_to_filter.pkl
            try:
                neighbour_emb = unique_span_dict[neighbour]
            except KeyError:
                # are these the neighbours that would/should be filtered?
                print(f"Potential neighbour '{neighbour}' no embedding found in unique_span_dict")
                continue

            # If the cosine similarity is below a given threshold, discard
            if (1 - cosine(self.embedding, neighbour_emb)) < cosine_sim_threshold:
                continue

            all_top_terms.append(neighbour)
            if len(all_top_terms) > top_k:
                break

        return all_top_terms[1:], inflections[1:]
