#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the implementation of a hierarchical clustering class based on modules
in scipy.cluster.hierarchy.

Author: Xian Lai
Date: Apr.14, 2017
"""

import pandas as pd
import numpy as np
import pickle
import geopandas as gpd
import multiprocessing as mp
from sklearn.model_selection import ParameterGrid
import Visual_static as vis
from InterpolateClustering import InterpolateClustering as IC
from bokeh import palettes
from bokeh.layouts import gridplot, row, column
from bokeh.plotting import figure
from bokeh.io import output_file, show, output_notebook
from bokeh.tile_providers import CARTODBPOSITRON_RETINA
from bokeh.models import GeoJSONDataSource, LinearColorMapper, LinearColorMapper, ColorBar, FuncTickFormatter

rank_func = lambda sr: sr.rank(method='min', ascending=False)
score_func = lambda sr: (sr - sr.min())/(sr.max() - sr.min())
score_rank_func = lambda sr: score_func(sr)/rank_func(sr)
l1_dist = lambda x: (abs(x[0] - lng) + abs(x[1] - lat))
scale_by_col = lambda grp: grp.apply(score_func, axis=0)

def base_plot(plot_width, plot_height, x_range, y_range, axis_visible=True):
    """ prepare the figure for plotting
    """
    p = figure(
        tools="pan,wheel_zoom,box_zoom,reset,save",
        plot_width=plot_width, 
        plot_height=plot_height,
        x_range=x_range, 
        y_range=y_range
    )
    if not axis_visible: p.axis.visible = False

    return p

def plot_rsg(stats, figure_size=400, silent=False):
    """ plot rank_score_graph
    """
    p = figure(plot_width=figure_size, plot_height=figure_size, title="Rank-Score Graph")
    if not silent:
        p.xaxis[0].ticker.desired_num_ticks = 20
        p.xaxis.axis_label = "ranking"
        p.yaxis.axis_label = "scoring"
        p.xgrid.minor_grid_line_color = 'black'
        p.xgrid.minor_grid_line_alpha = 0.1
    else:
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.legend.location = None
        p.xgrid.minor_grid_line_alpha = 0
        
    for cnt, col in enumerate(stats.columns):
        criterion = score_rank_func(stats[col])\
            .sort_values(ascending=False)\
            .reset_index(drop=True)
        p.line(
            criterion.index, criterion, line_width=2,
            color=palettes.Spectral11[2*cnt], alpha=0.3, legend=col
        )
        p.circle(
            criterion.index, criterion, line_width=2,
            color=palettes.Spectral11[2*cnt], alpha=0.8, legend=col
        )
    p.legend.click_policy="hide"
    
    if silent: return p
    else: show(p)


def compare_rsg(stats_set, n_cols):
    """
    """
    def _graph(stats):
        """
        """
        p = figure()
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.legend.location = None
        p.xgrid.minor_grid_line_alpha = 0
        p.xgrid.grid_line_alpha = 0

        for cnt, col in enumerate(stats.columns):
            criterion = score_rank_func(stats[col])\
                .sort_values(ascending=False)\
                .reset_index(drop=True)
            p.line(
                criterion.index, criterion, line_width=2,
                color=palettes.Spectral11[2*cnt], alpha=0.3
            )
            p.circle(
                criterion.index, criterion, line_width=2,
                color=palettes.Spectral11[2*cnt], alpha=0.8
            )

        return p
    ps = [_graph(stats) for stats in stats_set]
    layout = gridplot(ps, plot_width=200, plot_height=200, ncols=n_cols, toobar_location = None)
    show(layout)


def plot_patches(p, geojson_, alpha=0.5, silent=False):

    """ plot the grids on Google map.palettes.RdYlGn11

    Args:
        values (array): either the input data or output data as a 1-d array
        grids (list): the bounds of each grid

    """
    # p = base_plot(
    #     plot_width=1200, plot_height=1200, x_range=(-8237000,-8225000), 
    #     y_range=(4975000,4980000), axis_visible=False
    # )
    # output_file("plot_grids.html")
    geo_source = GeoJSONDataSource(geojson=geojson_)
    p.add_tile(CARTODBPOSITRON_RETINA)
    p.patches(
        'xs', 'ys', 
        fill_alpha=alpha, 
        fill_color='color', 
        line_color='navy', 
        line_width=0.5, 
        source=geo_source
    )
    
    if silent: return p
    else: show(p)


def plot_scatter(p, x, y, marker='x', size=15, alpha=0.5, line_color=None, 
                 fill_color="orange", silent=False):
    """
    """
    p.scatter(
        x, y, marker=marker, size=size, alpha=alpha, line_color=line_color, 
        fill_color=fill_color, 
    )

    if silent: return p
    else: show(p)


def compare_auc(df_auc, fw=800, fh=400, title=None):
    p = figure(plot_width=fw, plot_height=fh, title=title)
    p.xaxis.axis_label="sampling"
    p.yaxis.axis_label="area_under_curve"
    for cnt, col in enumerate(df_auc.columns):
        sr = df_auc[col]
        p.line(
            sr.index, sr, line_width=2,
            color=palettes.Spectral11[2*cnt], alpha=0.3, legend=col
        )
        p.circle(
            sr.index, sr, line_width=2,
            color=palettes.Spectral11[2*cnt], alpha=0.8, legend=col
        )
    p.legend.click_policy="hide"
    show(p)


def plot_image(arr, title=None, px_w=5, px_h=5, silent=False):
    """Plot image using given array.
    
    Args:
        arr: the np.array as pixel color coding
        px_w: a integer, the width of each pixel
        px_h: a integer, the height of each pixel
    """
    h, w = arr.shape
    p = figure(
        title=title, 
        plot_width=px_w * w, plot_height=px_h * h, 
        x_range=(0, w), y_range=(0, h),
        toolbar_location=None,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    p.xaxis.axis_label="sampling"
    p.yaxis.axis_label="combined_score"
    p.toolbar.logo = None
    cm = LinearColorMapper(palette=palettes.YlGn9)
    r = p.image(
        image=[arr], 
        x=0, y=0, dw=w, dh=h,
        color_mapper=cm,
    )
    color_bar = ColorBar(
        color_mapper=cm, 
        label_standoff=12, border_line_color=None, location=(0,0)
    )
    p.add_layout(color_bar, 'right')
    
    if silent: return p, r
    else: show(p)



