"""
Everything related to plotting data
"""

import matplotlib.pyplot as plt
import os


def plot_attributes_2d(X, y, C, classNames, attributeNames, i=0, j=1, saveFigure=False):
    """
    Method for plotting different attributes against each other
    :param X:
    :param y:
    :param C:
    :param classNames:
    :param attributeNames:
    :param i:
    :param j:
    :return:
    """
    # Plotting the data set (different attributes to be specified)
    plt.figure()

    for c in range(C):
        # select indices belonging to class c:
        class_mask = y==c
        plt.plot(X[class_mask,i], X[class_mask,j], 'o')

    plt.legend(classNames)
    svi_legend = ['SVI 0', 'SVI 1']
    plt.legend(svi_legend, loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=1)
    plt.rc('legend',fontsize=24)
    axis_font = {'size':'24'}
    plt.xlabel(attributeNames[i], **axis_font)
    plt.ylabel(attributeNames[j], **axis_font)
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)

    if saveFigure == True:
        plt.savefig('./ML_fall2018/Figures/' + attributeNames[i] +  "_vs_" + attributeNames[j]+".png")

    # Output result to screen
    plt.show()