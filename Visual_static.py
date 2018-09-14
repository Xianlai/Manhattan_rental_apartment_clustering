#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Author: Xian Lai
Date: Jul.21, 2017
"""

from bokeh.plotting import figure
from bokeh.palettes import Spectral11, RdYlGn11
from bokeh.io import show as bokehShow
from bokeh.models import ColumnDataSource
from scipy.cluster.hierarchy import dendrogram
import numpy as np

grey      = {'light':'#efefef', 'median':'#aaaaaa', 'dark':'#282828'}
discColor = {'blue':'#448afc', 'red':'#ed6a6a', 'green':'#80f442'}
cms       = {'spectral':Spectral11, 'RdYlGn':RdYlGn11}
dotSize   = 0.5
TOOLS     = 'pan, wheel_zoom, reset, save'


class Visual():
    
    def __init__(self, 
            figWidth = 500, figHeight = 300, 
            x_range=None, y_range=None,
            title="Clustering Plot"):
        """ 
        """
        
        self._basePlot(figWidth, figHeight, x_range, y_range, title)
        self._setFig()


    def _basePlot(self, figWidth, figHeight, x_range, y_range, title):
        """ set up a bokeh figure with given figure size and tools.
        """
        self.p = figure(
            plot_width=figWidth, 
            plot_height=figHeight, 
            title=title,
            x_range=x_range, 
            y_range=y_range,
            tools=TOOLS,
            toolbar_location='below',
            toolbar_sticky=False
        )


    def _setFig(self):
        """  set attributes of given bokeh fig
        """
        self.p.background_fill_color       = 'black'
        self.p.axis.visible                = False
        self.p.xgrid.grid_line_color       = None
        self.p.ygrid.grid_line_color       = None
        self.p.xgrid.minor_grid_line_color = None


    def plotClustering(self, data, color='mean', bins=None,
            orientation='portrait', show=True):
        """
        """
        self.data = data.copy()
        source = self._createCDS(self.data, color, bins)
        self._makePlot(source, orientation)
        if show: bokehShow(self.p)


    def _createCDS(self, data, color, bins):
        """ filter out the clusters we would like to show using masks 
        generated from range sliders.
        """
        # set up the plotting color and alpha for each cluster based on the 
        # column name chosen in select widge.
        # Divide the values of series into 11 bins(the length of 
        # discrete color map). And then map the values to their bin indices.
        if bins is None:
            ubound  = data[color].quantile(0.99)
            lbound  = data[color].quantile(0.01)
            data    = data[(data[color]>lbound) & (data[color]<ubound)]
            _, bins = np.histogram(data[color], 10)
        indices       = np.digitize(data[color], bins)
        data['color'] = [cms['RdYlGn'][i - 1] for i in indices]
        data['alpha'] = [i/2+0.3 for i in indices]
        self.bins     = bins

        return ColumnDataSource(data=data)


    def _makePlot(self, source, orientation):
        if orientation == 'portrait':
            self.p.plot_height = int(self.p.plot_width / 0.2777)
            self.r = self.p.scatter(
                x='x', y='y', color='color', alpha='alpha', 
                source=source, size=dotSize
            )
        else:
            self.p.plot_height = int(self.p.plot_width * 0.2777)
            self.r = self.p.scatter(
                x='y', y='x', color='color', alpha='alpha', 
                source=source, size=dotSize
            )









        
        


