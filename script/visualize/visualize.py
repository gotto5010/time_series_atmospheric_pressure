
import numpy as np
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm

execute_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.normpath(os.path.join(execute_dir, '..', '..'))
output_dir = os.path.normpath(os.path.join(project_root, 'result', 'visualize'))
raw_data_dir = os.path.normpath(os.path.join(project_root, 'data', 'raw'))

df = pd.read_csv(os.path.join(raw_data_dir, 'hourly_atmospheric.csv'))
df['Date'] = pd.to_datetime(df['Date'])

def plot_hourly_atmospherec():
    """
    plot hourly atmospherec
    :return: plot png
    """
    plt.figure(figsize=(15, 7))
    left = df['Date']
    height = df['atmospherec']
    plt.plot(left, height)
    plt.title('atmospheric per hour')
    plt.ylabel('atmospherec')
    plt.xlabel('date')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'plot_hourly_atmospherec.png'))

def plot_twotype_color(df):
    df = df.set_index('Date')
    plt.figure(figsize=(15, 7))
    N = datetime(2018, 5, 1)
    ax = df['atmospherec'].plot()
    df.loc[df.index <= N, 'atmospherec'].plot(color='r', ax=ax)
    ax = df.plot(x=df.index, y='atmospherec')
    plt.title('plot hourly atmospherec')
    plt.ylabel('atmospherec')
    plt.savefig(os.path.join(output_dir, 'plot_twotype_atmospherec.png'))
    plt.show()

def plot_month_std_atmospherec(df):
    """
    plot std atmospherec per month
    :param df:
    :return:
    """
    df = df.set_index('Date')
    month_df = df.resample('M').std()
    month_df = month_df.reset_index()
    left = month_df['Date']
    height = month_df['atmospherec']
    plt.plot(left, height)
    plt.title('month std of atmospherec')
    plt.xlabel('month')
    plt.ylabel('standard deviation')
    plt.savefig(os.path.join(output_dir, 'mont_std_atmospherec.png'))

def scatter_atmospherec():
    lag_df = df.copy()
    lag_df['lag_atmosperec'] = lag_df['atmospherec'].shift(1)
    lag_df = lag_df.dropna()

    plt.figure(figsize=(8, 6))
    x = lag_df['lag_atmosperec']
    y = lag_df['atmospherec']
    plt.scatter(x, y)
    plt.title('scatter atmospherec vs lag_atmospherec')
    plt.ylabel('asmospherec')
    plt.xlabel('lag_asmospherec')
    plt.savefig(os.path.join(output_dir, 'scatter_atmospherec_vs_lag_atmospherec.png'))

ts = df['atmospherec']
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(ts, lags=50, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(ts, lags=50, ax=ax2)
plt.savefig(os.path.join(output_dir, 'acf.png'))
