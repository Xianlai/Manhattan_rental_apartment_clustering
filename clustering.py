#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:37:23 2017

@author: LAI
"""
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt


class Clustering():
    
    def __init__(self, df, method='average', metric='cityblock'):
        
        self.dfToMatrix(df)
        self.method = method
        self.metric = metric
        self.Z = linkage(self.data, method=self.method, metric=self.metric)
        
        
    def dfToMatrix(self, df):
        df = (df - df.mean()) / df.std()
        self.data = df.as_matrix()
        
    
    def getClusters(self, k=600):

        clusters = fcluster(self.Z, k, criterion='maxclust')
        self.clusters = clusters.reshape(clusters.size, 1)
        
    def getClusterPopularity(self):
        """docstring for getClusterPopularity"""
        data_clustered = np.concatenate((self.data, self.clusters), axis=1)
        df_cluster = pd.DataFrame(data_clustered, columns=['x', 'y', 'popularity','cluster'])
        
        gb = df_cluster.groupby('cluster')
        cluster_mean = gb['popularity'].mean().reset_index()
        cluster_mean['cluster_popularity'] = standardScale(cluster_mean['popularity'])
        del cluster_mean['popularity']
        df_cluster_new = df_cluster.merge(cluster_mean, how='left', on='cluster')

        return df_cluster_new
        

    def clusterStats(self):
        
        data_clustered = np.concatenate((self.data, self.clusters), axis=1)
        df_cluster = pd.DataFrame(data_clustered, columns=['x', 'y', 'popularity','cluster'])
        df_cluster['x+y'] = df_cluster['x'] + df_cluster['y']

        gb = df_cluster.groupby('cluster')
    
        sz = gb['popularity'].size()
        
        ## what are the sizes of clusters
        percentile_15 = sz.quantile(0.15)
        percentile_100 = sz.quantile(1)
        num_singlton = len(sz[sz==1])
        
        def clusterArea(df):
            x_span = df['x'].max() - df['x'].min() 
            y_span = df['y'].max() - df['y'].min()
            return x_span * y_span
        
        areas = gb.apply(clusterArea)
        area_85 = areas.quantile(0.85)
        
        ## what is the inter-cluster variance
        Var = gb['popularity'].var()
        interVar = Var.mean()
        
        self.nearestK = nearestKClusters(gb)
        
        ## what is the intra-cluster variance
        Mean = gb['popularity'].mean()
        
        intraVars = [Mean.loc[self.nearestK[i]].var() for i in Mean.index]
        
        intraVar = sum(intraVars)/len(intraVars)
        
        return percentile_15, -percentile_100, -num_singlton, -area_85, -interVar, intraVar
        
        
    def truncated_dendrogram(self, lastp, figsize=(10, 15)):
        plt.figure(figsize=figsize)
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        
        dendrogram(
            self.Z,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=lastp,  # show only the last p merged clusters
            leaf_rotation=90.,
            leaf_font_size=0.,
            show_contracted=True)  # to get a distribution impression in truncated branches
        
        plt.show()
    

def nearestKClusters(gb, k=5):
    centroids = gb['x+y'].mean()
    nearestK = {}
    for name, group in gb:
        dists = centroids - centroids.loc[name]
        dists = dists.abs()
        nearestK[name] = dists.sort_values(ascending=True).head(n=k).index
    return nearestK
    
def standardScale(sr):
    """docstring for standardScale"""
    sr = (sr - sr.min()) / (sr.max() - sr.min())
    return sr

