from typing import Dict, List, Any
import os, glob, pickle, json, concurrent
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import silhouette_score


class ElbowAndSilhouette:
    def __init__(self, cluster_dir, elbow=True, silhouette=True):
        self.sum_of_squared_distances = []
        self.silhouette_avg_euclidean = []
        self.silhouette_avg_correlation = []
        self.elbow = elbow
        self.silhouette = silhouette
        
        self.cluster_dir = cluster_dir
        self.cluster_spans = pickle.load(open(cluster_dir + "unique_spans.pkl", 'rb'))
        self.cluster_data = pickle.load(open(cluster_dir + "unique_embeddings.pkl", 'rb'))

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

    def compute_scores_for_single_model(self, pkl_name):
        centroids, assignments = pickle.load(open(pkl_name, 'rb'))

        # Drop NAN clusters that 'died' if they exist
        mask = ~np.isnan(centroids).any(axis=1)
        for idx, m in enumerate(mask):
            if not m:
                # placeholder so we don't have to deal with shifting indices of assigned clusters (out-of-index)
                centroids[idx] = np.array([0.0001]*768)

        num_clusters = len(centroids)

        if self.silhouette and num_clusters > 8000:
            print("""
                For large numbers of clusters (e.g. > 8K), computing the silhouette score can run into memory
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
            self.silhouette_avg_correlation.append(silhouette_score(self.cluster_data, 
                                                               assignments, metric="correlation"))


    def save_and_plot_with_scores(self, clustering_type, pkl_files):
        print("Plotting the figure")
        # ensure order is based on the number of clusters
        dict_with_values = {}
        for idx, x in enumerate(pkl_files):
            try:
                num_clusters = int(x.split('_')[1])
                dict_with_values[num_clusters] = {
                                           "num_clusters": num_clusters,
                                           "sum_of_squared_distances": self.sum_of_squared_distances[idx],
                                           "silhouette_avg_euclidean": self.silhouette_avg_euclidean[idx],
                                           "silhouette_avg_correlation": self.silhouette_avg_correlation[idx]
                }
            except:
                continue

        dict_to_plot = {
               "num_clusters": [],
               "sum_of_squared_distances": [],
               "silhouette_avg_euclidean": [],
               "silhouette_avg_correlation": []
            }
        ordered_keys = [k for k in dict_with_values.keys()]
        ordered_keys.sort()
        for k in ordered_keys:
            for v in dict_with_values[k]:
                dict_to_plot[v].append(dict_with_values[k][v])

        # create a pandas DataFrame that we'll save and plot
        df_to_plot = pd.DataFrame(dict_to_plot)
        # save 
        name_for_file = self.cluster_dir + clustering_type + "_" + \
                        "_".join([str(n) for n in dict_to_plot['num_clusters']]) + ".csv"
        df_to_plot.to_csv(name_for_file)
        # plot
        sn.set_style("darkgrid", {"axes.facecolor": ".95"})
        df_to_plot.set_index('num_clusters').plot(kind='line', 
                                                  color=['seagreen', 'salmon', 'red'],
                                                  fontsize=12,
                                                  subplots=True)


    def compute_scores_for_models(self, clustering_type, pkl_files):

        print("Computing elbow and silhouette (if not too many num_clusters) scores.")
        csv_file = max(glob.glob(self.cluster_dir + clustering_type + "*.csv"))
        latest_df = pd.read_csv(csv_file)
        updated_pkl_files = []
        for pkl_name in tqdm(pkl_files):
            if clustering_type not in pkl_name:
                print(f"{pkl_name} is not of expected type? {clustering_type}")
                continue

            num_clusters = pkl_name.split('_')[1]
            try:
                int(num_clusters)
            except:
                print(f"File may not have the right naming format: {pkl_name}")
                continue

            updated_pkl_files.append(pkl_name)
            if num_clusters in csv_file:
                print(f"Loading values from existing csv file: {pkl_name}")
                # already computed so we'll reuse the values
                temp = latest_df.loc[latest_df['num_clusters'] == int(num_clusters)]
                self.sum_of_squared_distances.append(temp['sum_of_squared_distances'].tolist()[0])
                self.silhouette_avg_euclidean.append(temp['silhouette_avg_euclidean'].tolist()[0])
                self.silhouette_avg_correlation.append(temp['silhouette_avg_correlation'].tolist()[0])
                continue

            # compute value if needed
            print(f"Working on: {pkl_name}")
            self.compute_scores_for_single_model(pkl_name)
            
        self.save_and_plot_with_scores(clustering_type, updated_pkl_files)



class ClusterDict:
    def __init__(self, 
                 ref_corp_unique,
                 unique_spans, 
                 unique_clustering_data, 
                 centroids, 
                 assignments,
                 embedding_fp="output/"
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

    def compute_span_dict(self, unique_spans: [str], max_num_cpu_threads: int = 8) -> Dict[str, Any]:
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
            if span['label'] not in cluster_dict.keys():
                cluster_dict[span['label']] = [[span['distance'], span['span']]]
            else:
                cluster_dict[span['label']].append([span['distance'], span['span']])

        # reorder list of terms per cluster, based on distance
        for cluster_id in cluster_dict.keys():
            cluster_dict[cluster_id] = sorted(cluster_dict[cluster_id], key=lambda s: (s[0]))
        return cluster_dict

    def prep_cluster_dict(self, chosen_num_clusters:int):
        """
        Loads or prepares the dictionary that holds information on clusters; spans and their respective distances
        to the centroid of a cluster. A file called `phrase_cluster_dict.json`, always stored at `self.embedding_fp`.

        :return phrase_cluster_dict:   Dictionary holding `{cluster ID: [[distance, span], ...]}`.
        """
        cluster_dict_filepath = self.embedding_fp + f"cluster_dict_{chosen_num_clusters}.json"
        if os.path.exists(cluster_dict_filepath):
            print("Loading a pre-computed cluster dictionary from file.")
            phrase_cluster_dict = json.load(open(cluster_dict_filepath, 'r'))
            clusters_to_filter = set(pickle.load(open(self.embedding_fp + "clusters_to_filter.pkl", 'rb')))
        else:
            print("Computing the cluster dictionary.")
            span_dict = self.compute_span_dict(self.unique_spans)
            phrase_cluster_dict = self.convert_to_cluster_dict(span_dict)
            with open(cluster_dict_filepath, 'w') as f:
                json.dump(phrase_cluster_dict, f)

            unwanted_terms = set(self.ref_corp_unique)
            clusters_to_filter = []
            for c_id, dists_and_terms in phrase_cluster_dict.items():
                term_list = [t for _, t in dists_and_terms]
                unwanted_terms_in_cluster = [t for t in term_list if t in unwanted_terms]
                # for our corpus and cluster_data for filtering, a count of 1 unwanted term seems to work well
                if len(unwanted_terms_in_cluster) > 0:      # todo this was set to 2 before, would need to re-run
                    clusters_to_filter.append(c_id)

            # store the clusters_to_keep for filtering later on
            with open(self.embedding_fp + "clusters_to_filter.pkl", 'wb') as f:
                pickle.dump(clusters_to_filter, f)

        return phrase_cluster_dict, clusters_to_filter