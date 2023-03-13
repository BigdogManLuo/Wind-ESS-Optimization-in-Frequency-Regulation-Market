import pandas as pd
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt

wind_dataset=pd.read_excel("data/wind/west_wind_farm.xlsx")    #风电最大出力

#%%
windPowerData=wind_dataset.values[10:33+1,2]

#%%

def Interpolation(windPowerData,num):

    x=np.linspace(0,windPowerData.shape[0],windPowerData.shape[0])  #原数据 
    x_new=np.linspace(0,windPowerData.shape[0],1800*24)   #新数据
    
    f=interpolate.interp1d(x,windPowerData,kind="cubic")#强制通过所有点(x,windPowerData,kind='quadratic')
    y_new=f(x_new)
    
    return y_new

num=1800*24
y_new=Interpolation(windPowerData,num)

#%%
plt.subplot(1,2,1)
plt.scatter(np.arange(0,24),windPowerData)
plt.subplot(1,2,2)
plt.plot(y_new)