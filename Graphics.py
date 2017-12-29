import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import os

color = ["b", "r",]

# @listFile is a vector of files names ['file1','file2',...]
# If @exportPlotInPNG is True, function save the plot and doesn't show the graphic
# The first line in the file should be the legend that you want see at plot
def plotLearningCurve(listFile, name="",exportPlotInPNG=False, c=0):

    def readFile(fileName):
        costs = []
        f = open(fileName, 'r')
        legend = f.readline().split('\n')[0]
        for line in f:
            costs.append(float(line.split('\n')[0]))
        f.close()
        return costs,legend

    if (len(listFile)>7):
        print "Too many files."
        return

    if (len(listFile)==0):
        print "EMPTY VECTOR!, I need at least one file."
        return

    plt.figure(num=None, figsize=(15, 8), dpi=60, facecolor='w', edgecolor='k')
    plt.xlabel("Iterations")
    plt.ylabel(name)
    plt.title("Learning")
    for i in range(len(listFile)):
        costs,legend = readFile(listFile[i])
        plt.plot(costs,color[i]+'-',label=legend,linewidth=2.0)
        plt.text(np.argmax(costs), np.max(costs) + 0.02, 'Max = '+str(np.max(costs)), fontsize = 10, color=color[c],
                 horizontalalignment='center', verticalalignment='center')
        plt.text(np.argmin(costs), np.min(costs) - 0.04, 'Min = '+str(np.min(costs)), fontsize = 10, color=color[c],
                 horizontalalignment='center', verticalalignment='center')
    plt.legend(loc="upper center")
    plt.grid(True)
    if (exportPlotInPNG):
        plt.savefig(str(name)+".png")
    else:
        plt.show()
