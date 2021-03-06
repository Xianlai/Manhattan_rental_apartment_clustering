#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the implementation of a hierarchical clustering class based on modules
in scipy.cluster.hierarchy.

Author: Xian Lai
Date: Apr.14, 2017
"""

from helpers import *
from scipy.cluster.hierarchy import linkage, fcluster


class InterpolateClustering():

    """
    This is the class for a hierarchical clustering that we can query it with 
    different cutting level of hierarchy and different filtering conditions.

    Input:
        - a listing dataset and a building dataset.
        - the hyperparameters for clustering model: n_ngb, IDWpower, method 
        and metric.

    Clustering process:
        - Data preparing: retain the data matrix for clustering. The data is 
          the building dataset with popularity interpolated from listing 
          dataset using inverse distance weighting.

        - Clustering: perform hierarchical clustering with the interpolated 
          data and given model hyperparameters.

    Fetching clusters:
    After the clustering process is done, we can query the information we need 
    and fetch or plot out the corresponding clusters.

        - n_clusters: the cutting level of dendrogram, namely the number of 
          clusters.

        - clusterStats: we can filter or sort the clusters by their statistics
          like the average popularity or popularity variance, number of 
          buildings inside cluster etc.
    """

    def __init__(self, df_listing, df_blocks):
        """
        """
        self.listings = df_listing  # The dataframe containing listing data
        self.blocks   = df_blocks   # The dataframe containing building data

    def analysis(self, IDWpower=1, n_clusters=600, method='average', 
                 metric='cityblock', rad=0.01):
        """ prepare the building dataset and perform clustering.
        """
        self.prepare_data(IDWpower, rad)
        self.clustering(n_clusters, method, metric)


    #-------------------------- DATA PREPARING -------------------------------
    def prepare_data(self, IDWpower, rad):
        """ Interpolate the popularity values for each building in the build-
        ing dataset and whitten the data.
        """
        blocks = self.blocks.copy()
        blocks['interest_level'] = blocks['centroid']\
            .apply(lambda x: self._interpolate(x, IDWpower, rad))
        blocks = blocks.drop('centroid', axis=1)
        self.data = self._whitening(blocks).values

        # print("Interpolation done.")


    def _whitening(self, df):
        """ Whiten the data by center it and scale the std to 1.
        """
        return (df - df.mean()) / df.std()


    def _interpolate(self, block, IDWpower, rad):
        """ calculate the manhattan distance between each building in listing
        dataset and the given target building. If there are buildings in the
        listing dataset in the same location as the building dataset, take the
        mean of popularities of overlaying buildings as the popularity value 
        of target building. If not, then use inverse distance weighting to 
        interpolate the popularity values using the nearest neighbors.

        inputs:
        ------
        - building: The target build without popularity value.

        output:
        -------
        - value: The interpolated popularity value.
        """
        lng, lat = block
        mhtDist = lambda x: (abs(x[0] - lng) + abs(x[1] - lat))
        
        ngbrs = self.listings[
            ['x', 'y', 'interest_level', 'centroid', 'listing_time']
        ]
        ngbrs = ngbrs[(ngbrs['x'] > (lng - rad)) & (ngbrs['x'] < (lng + rad))]
        ngbrs = ngbrs[(ngbrs['y'] > (lat - rad)) & (ngbrs['y'] < (lat + rad))]
        ngbrs['dist'] = ngbrs['centroid'].apply(mhtDist)

        dists     = ngbrs['dist']
        t_past    = ngbrs['listing_time']
        interests = ngbrs['interest_level']
        overlay   = interests[dists <= 0.0005]
        
        if not overlay.empty:
            return overlay.mean()
        else:
            weights = 1 / (dists**IDWpower) * t_past
            weights = weights / weights.sum()
            return sum(interests * weights)

    
    #---------------------------- CLUSTERING ---------------------------------
    def clustering(self, n_clusters=600, method='average', metric='cityblock'):
        """ Perform clustering with prepared data and given method, metric. 
        Fetch the cluster index of each building in self.data and combine them
        as self.clusters.

        clusters: each block[x, y, popularity, cluster_idx]
        """
        self.method   = method     # The method to calculate distance
        self.metric   = metric     # The metric to define distance
        self.linkage  = linkage(
            self.data, method=self.method, metric=self.metric
        )
        clusters = fcluster(
            self.linkage, n_clusters, criterion='maxclust'
        )
        clusterIdx    = clusters.reshape(clusters.size, 1)
        clusters      = np.concatenate((self.data, clusterIdx), axis=1)
        self.clusters = pd.DataFrame(
            clusters, columns=['x', 'y', 'popularity','cluster']
        )
        self._get_clusterStats()
        self._get_clusteringStats()
        self._joinStats()

        # print("Clustering done.")


    def _get_clusterStats(self):
        """ Calculate the statistics for each cluster. The stats inlcude:
        - 'mean'     : cluster popularity mean
        - 'variance' : cluster popularity variance
        - 'size'     : cluster size(number of buildings inside)
        - 'area'     : cluster area(the area of bounding box)
        
        """
        boundingBoxArea = lambda c: \
            (c['x'].max() - c['x'].min()) * (c['y'].max() - c['y'].min())

        grps = self.clusters.groupby('cluster')
        idx  = grps.groups.keys()

        mean = grps['popularity'].mean()
        var  = grps['popularity'].var()
        size = grps['popularity'].size()
        area = grps.apply(boundingBoxArea)

        self.clusterStats = pd.DataFrame({
            'mean':mean, 'variance':var, 'size':size, 'area':area
        })


    def _get_clusteringStats(self):
        """ Calculate the 6 statistics for the evaluation of clustering:
        stats = {
            'lgClusterSize' : the size of top 15% largest cluster,
            'smClusterSize' : the size of top 15% smallest cluster,
            'n_singlton'    : the number of singleton clusters,
            'lgClusterArea' : the area of top 15% largest cluster,
            'interVariance' : the inter cluster variance,
            'intraVariance' : the intra cluster variance,
        }
        (The good clustering requires a balanced grouping, so we want to make 
                these statistics all varying in the same way.!!!!)
        """
        stats = {}
        sizes = self.clusterStats['size']
        stats['lgClusterSize'] = -sizes.quantile(0.85)
        stats['smClusterSize'] = sizes.quantile(0.15)
        stats['n_singlton']    = -len(sizes[sizes==1])
        stats['lgClusterArea'] = -self.clusterStats['area'].quantile(0.85)
        stats['interVariance'] = -self.clusterStats['variance'].mean()
        stats['intraVariance'] = self.clusterStats['mean'].var()

        self.clusteringStats   = stats
        return stats


    #----------------------------- QUERYING ----------------------------------
    def fitering(self, mask):
        """ filter the buildings with given cluster mask.

        inputs:
        -------
        - mask: the indices of clusters we want to fetch.
        """
        buildings = self.clusters.copy()
        return buildings[buildings['cluster'].isin(mask)]


    def _joinStats(self,):
        """ join the cluster data with its corresponding standardized cluster 
        statistics.
        """
        df = self.clusterStats.copy()
        df = (df - df.min()) / (df.max() - df.min())
        self.clusters = self.clusters\
            .merge(df.reset_index(), how='left', on='cluster')










