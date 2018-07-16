import matplotlib
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import pandas as pd

from datetime import date
df = pd.read_csv('mydata/beijingpm25.csv', parse_dates={'date':['year', 'month', 'day']}, index_col='No')

# correlation
df.corr()

# correlation chart
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('BEIJING PM2.5')
    labels=list(df1)
    ax1.set_xticklabels(labels,fontsize=12)
    ax1.set_yticklabels(labels,fontsize=12)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[-1.,-.75,-.5,0,.5,.75,.8,.9,1])
    plt.show()

correlation_matrix(df)

# scatter
df.plot(kind="scatter", x="Iws", y="pm2.5")

# 3D scatter
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["TEMP"],df["PRES"],df["pm2.5"])

