#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of interactive plotting of Manhattan rental apartment
clustering on Bokeh server.

Author: Xian Lai
Date: Nov.30, 2017
"""

import pandas as pd
import numpy as np

from bokeh.layouts import widgetbox, row
from bokeh.palettes import Spectral11, RdYlGn11
from bokeh.plotting import curdoc, figure
from bokeh.models.widgets import RangeSlider, Select
from bokeh.models import ColumnDataSource

cms       = {'Spectral':Spectral11, 'RdYlGn':RdYlGn11}
dotSize   = 0.5
TOOLS     = 'pan, wheel_zoom, reset, save'
data      = pd.read_csv("data/clustering_data.csv")
stats     = [
    "Popularity Mean", "Popularity Variance", "Cluster Size", "Cluster Area"
]


class InteractivePlot():

    def __init__(self, figWidth=350, figHeight=1260):
        """
        inputs:
        -------
        - figWidth: the width of plotting figure
        - figHeight: the height of plotting figure
        """
        self.figWidth  = figWidth
        self.figHeight = figHeight

        self._makeCDS(data, stats[0])  # generate column data source
        self._makeFig()              # make the fig and scatter plot
        self._setFig()               # set the properties of this fig
        self._setWidgets()           # set up widgets and their callback funcs

        # push the widget box and figure to the bokeh server
        curdoc().add_root(row(self.widgets, self.p)) 
        curdoc().title = "interactive clustering of Manhattan rental apts"


    def _makeFig(self,):
        """ set up a bokeh figure with given figure size and tools.
        """
        # set up a figure with given figure size and tools
        self.p = figure(
            title="Clustering Plotting",
            plot_width=self.figWidth, 
            plot_height=self.figHeight,
            tools=TOOLS
        )
        # add box zoom tool with restriction on width
        # add scatter plot in the fig using current data source
        self.r = self.p.scatter(
            x='x', y='y', color='color', alpha='alpha', 
            source=self.source, size=dotSize
        )
        # assign the data_source of scatter plot to a variable for later update
        self.ds = self.r.data_source


    def _setFig(self,):
        """ set attributes of given bokeh fig
        """
        self.p.background_fill_color       = 'black'
        self.p.axis.visible                = False
        self.p.xgrid.grid_line_color       = None
        self.p.ygrid.grid_line_color       = None
        self.p.xgrid.minor_grid_line_color = None


    def _makeCDS(self, data, colorby):
        """ map the given column of values to colors and alphas and create a
        columnDataSource.

        inputs:
        -------
        - data: the input dataframe
        - colorby: the name of column we want to map to colors and alphas  
        """
        # remove outlier clusters
        ubound  = data[colorby].quantile(0.99)
        lbound  = data[colorby].quantile(0.01)
        data_new = data[(data[colorby]>lbound) & (data[colorby]<ubound)]
        # divide the colorby column into 10 bins(the length of discrete color 
        # map). And map the values to their bin indices.
        _, bins = np.histogram(data_new[colorby], 10)
        indices       = np.digitize(data_new[colorby], bins)
        # map the indices to the colors and alphas
        data_new['color'] = [cms['RdYlGn'][i - 1] for i in indices]
        data_new['alpha'] = [i/2+0.3 for i in indices]
        self.source   = ColumnDataSource(data=data_new)


    def _setWidgets(self,):
        """ set up the widgets used to control plotting. The range sliders will
        control the fitering of data points and the select drop down menu will
        control which column to map color from.

        They are all connected to the corresponding callback function so when 
        the widget is changing, the plotting will be updated.
        """
        param = {
            'start':0, 'end':1, 'value':(0,1), 'step':0.05
        }
        self.mean    = RangeSlider(title=stats[0], **param)
        self.var     = RangeSlider(title=stats[1], **param)
        self.size    = RangeSlider(title=stats[2], **param)
        self.area    = RangeSlider(title=stats[3], **param)
        self.colorby = Select(title="Colored by:", value=stats[0], options=stats)

        self.mean.on_change('value', self._sliderCallback)
        self.var.on_change('value', self._sliderCallback)
        self.size.on_change('value', self._sliderCallback)
        self.area.on_change('value', self._sliderCallback)
        self.colorby.on_change('value', self._selectorCallback)

        self.widgets = widgetbox(
            self.colorby, self.mean, self.var, self.size, self.area, width=300
        )

    def _makeMask(self):
        """ make a mask of dataframe using values of sliders
        """
        f = lambda rng, ser: (ser > rng[0]) & (ser < rng[1])
        mask_1 = f(self.mean.value, data[self.mean.title])
        mask_2 = f(self.var.value, data[self.var.title])
        mask_3 = f(self.size.value, data[self.size.title])
        mask_4 = f(self.area.value, data[self.area.title])
        return mask_1 & mask_2 & mask_3 & mask_4


    def _sliderCallback(self, attr, old, new):
        """ the callback function for all sliders. makes update to the plotting
        data source.
        """
        mask = self._makeMask()
        data_new = data[mask].copy()
        self._makeCDS(data_new, self.colorby.value)
        self.ds.data = self.source.data

    
    def _selectorCallback(self, attr, old, new):
        self._makeCDS(data, self.colorby.value)
        self.ds.data = self.source.data


InteractivePlot()



