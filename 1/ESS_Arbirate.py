# -*- coding: utf-8 -*-
#%%
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import time
from numba import jit
import pandas as pd
import pickle


#状态转换方程
def update(state,action):
    
    """
    Parameters
    ----------
    state : np.array 2x1   时间和储能能量剩余
    action : float   1x1  充放功率

    Returns
    -------
    newState : np.array 2x1 新状态的时间和储能能量剩余
    
    """
    newState=np.array([0,0])
    #时间转换
    if state[0]<23:   #到下一个时刻
        newState[0]=state[0]+1
    else:
        newState[0]=state[0]
    #储能能量变化
    if action >=0:  #如果是放电
        newState[1]=state[1]-(action/eta_dch)*dt
    else:           #如果是充电
        newState[1]=state[1]-action*dt*eta_ch
        
    return newState

 
#状态-动作值函数定义
def q(state,action):
    
    """
    Parameters
    ----------
    state : np.array 2x1   时间和储能能量剩余
    action : float   1x1  充放功率

    Returns
    -------
    如果在state状态下action是可行的，那么正常计算状态-动作值函数
    如果是不可行的，那么返回一个很大的惩罚项

    """
    newState=update(state,action) #获取下一个状态
    if newState[1]>=0 and newState[1]<=E_max:  # 如果新状态是在可行域
        a=np.where(A==action)[0][0]  #获取当前action在状态空间的序号
        return r[state[0],state[1],a]+gamma*V[newState[0],newState[1]]
    else:
        return -100000

#%% 策略迭代算法 （未加速）   
"""
lastSumV=-1
count=0
#开始计时
start=time.time()
while(not(abs(lastSumV-V.sum())<=1e-3)):
    count+=1
    lastSumV=V.sum()
    #策略评估
    for t in range(S.shape[0]):  #遍历所有的状态
        for s in range(S.shape[1]):
            tmp=0
            state=np.array([t,s])
            for a in range(len(A)):  #遍历所有的动作
                newState=update(state,A[a])
                #判断是否超出可行域
                if newState[1]>=0 and newState[1] <= E_max:  #如果在可行域内
                    tmp+=pi[t,s,a]*(r[t,s,a]+gamma*V[newState[0],newState[1]])
            V[t,s]=tmp
    #策略改进
    for t in range(S.shape[0]):  #遍历所有的状态
        for s in range(S.shape[1]):
            state=np.array([t,s])
            maxq=-100000
            for a in range(len(A)):
                #先把所有的q(s,a)放进一个数组里面
                if q(state,A[a])>maxq:
                    maxq=q(state,A[a])
                    a_best=a
            #按照最大化q改善策略pi
            pi[t,s,:]=0
            pi[t,s,a_best]=1
  
#结束计时
end=time.time()
"""

#%%  值函数迭代算法（jit装饰器加速）

@jit(nopython=True)
def valueIteration(V,pi):
    newPi=pi
    newV=V
    for t in range(S.shape[0]):
        for s in range(S.shape[1]):
            state=np.array([t,s])
            maxQ=-10000000
            for a in range(A.size):
                newState=np.array([0,0])
                #时间转换
                if state[0]<23:   #到下一个时刻
                    newState[0]=state[0]+1
                else:
                    newState[0]=state[0]
                #储能能量变化
                action=A[a]
                if action >=0:  #如果是放电
                    newState[1]=state[1]-(action/eta_dch)*dt
                else:           #如果是充电
                    newState[1]=state[1]-action*dt*eta_ch
                #计算Q
                if newState[1]>=0 and newState[1]<=E_max:  # 如果新状态是在可行域
                    tmpQ= r[state[0],state[1],a]+gamma*V[newState[0],newState[1]]
                else:
                    tmpQ= -10000000
                if tmpQ>maxQ:
                    maxQ=tmpQ
                    a_best=a
            newPi[t,s,:]=0
            newPi[t,s,a_best]=1
            newV[t,s]=maxQ
    return newV,newPi
#%% 参数设定
if __name__=='__main__':
    

    
    #电池参数
    T=24 #运行周期
    dt=1 #单步时间
    E_max=360 #额定容量 100 kWh
    P_max=180 #最大充放功率 100 kW
    eta_ch=0.95 #电池效率
    eta_dch=0.95
    N_max=5000  #100%dod 最大循环次数
    C_in=10800000*6
    
    #学习率定义
    gamma=0.98
    #电价参数
    price_dataset=pd.read_excel("data/market price/From_IEEE2017.xlsx")   #能量、调频市场价格
    is_dollar=1 # 市场价格单位判断  0人名币 1 美元
    exchangeRate=6.86 #美元汇率
    price=np.float32(price_dataset.values[1::1,1])*(1+is_dollar*(exchangeRate-1)) #日前能量市场价格 ￥/MWh
    price=price*0.1
    #状态空间定义
    S=np.array([np.arange(0,E_max+1) for _ in range(T)]) #行：时间段  列：储能的能量剩余，颗粒度为1

    #动作空间定义
    A=np.arange(-P_max,P_max+1) #0~30分别表示储能的充放功率，正放负充，颗粒度为1
    
    #状态值函数
    V=np.zeros(S.shape,dtype=float)   #与S同维度
    #策略定义
    pi=np.ones((S.shape[0],S.shape[1],len(A)),dtype=float)*(1/len(A))   # T x S x A
    for s in range(S.shape[1]):
        for a in range(len(A)):
            #计算s,a对应的下一个状态
            if A[a]>=0:  #如果是放电动作
                s_new=s-(A[a]/eta_dch)*dt
            elif A[a]<0:  #如果是充电动作
                s_new=s-A[a]*eta_ch*dt
            #设置禁止域
            if s_new<0 or s_new>E_max:  #如果新状态超出了可行域
                pi[:,s,a]=0 #forbidden area

    #即时回报定义
    r=np.zeros((S.shape[0],S.shape[1],len(A)),dtype=float)
    for t in range(S.shape[0]):
        for s in range(S.shape[1]):
            for a in range(len(A)):
                #计算s,a对应的下一个状态
                if A[a]>=0:  #如果是放电动作
                    s_new=s-(A[a]/eta_dch)*dt
                elif A[a]<0:  #如果是充电动作
                    s_new=s-A[a]*eta_ch*dt
                if s_new>=0 and s_new<E_max:  #如果是在可行域
                    r[t,s,a]=price[t]*A[a]-C_in*(abs(s_new-s)/(2*N_max*E_max))  #能量收益-寿命折旧成本
                else:                         #否则给一个很大的惩罚项
                    r[t,s,a]=-10000000

#%% 值函数迭代
    lastSumV=-1
    count=0
    #开始计时
    start=time.time()
    while(not(abs(lastSumV-V.sum())<=1e-3)):
        count+=1
        lastSumV=V.sum()
        V,pi=valueIteration(V,pi)
    #结束计时
    end=time.time()


#%%实际运行测试
    
    #初始储能状态
    state=np.array([0,180])

    recordA=np.zeros((24,1),dtype=float)
    recordR=np.zeros(24,dtype=float)
    for t in range(24):
        #e=np.where(S[state[0],:]==round(state[1],-2))[0][0]
        a=np.argmax(pi[state[0],state[1],:])
        state=update(state,A[a])
        recordA[t]=A[a]
        
#%% 可视化
    plt.subplot(2,1,1)
    plt.step(np.arange(0,24),price,color="#2e86de")
    plt.xlabel("时间(hour)")
    plt.ylabel("电价(￥/0.1MWh)")
    plt.grid()
    
    plt.subplot(2,1,2)
    plt.step(np.arange(0,24),recordA,color="#ff9f43")
    plt.xlabel("时间(hour)")
    plt.ylabel("充放功率(0.1 MWh)")
    plt.grid()
    
    plt.show()
    
    
    print("总用时"+str(end-start)+"秒")
    print("储能累积收益："+str(np.dot(recordA[:,0],price))+"￥")
    
    pickle.dump(recordA,open("result/recordA.pkl","wb"))
