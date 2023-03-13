import pandas as pd
import numpy as np
import pyscipopt
import gc
from wind_interpolate import Interpolation
import pickle
#from FigurePlot import showPriceData,showWindData
#%% 算例数据加载
print("算例数据加载中...")
price_dataset=pd.read_excel("data/market price/From_IEEE2017.xlsx")   #能量、调频市场价格
wind_dataset=pd.read_excel("data/wind/west_wind_farm.xlsx")    #风电最大出力
regD_dataset=pd.read_excel("data/regD/regD.xlsx") #调频信号
print("算例数据加载完成。")

#%% 参数设定

#风电参数
Pw_max=36    #额定容量 MW  
numUnit=20   #风电机组数量
with open("data/D-1/W.pkl","rb") as f:
    W=pickle.load(f)  #整个风电场最大功率预测值 在实际值上叠加了均值为0，标准差为0.05的正态分布随机数
W_min=np.zeros(24,dtype='float32') #每小时的风电最大出力预测的最小值
for t in range(24):
    W_min[t]=W[t,:].min()
with open("data/D-1/eps_wind_reg.pkl","rb") as f:
    eps_wind_reg=pickle.load(f)   #风电调频误差  均值为0标准差为0.05的正态分布
climbingUpRate=0.2  #上爬坡率
climbingDownRate=0.15 #下爬坡率
maxPwUp=climbingUpRate*Pw_max  #1h的最大向上变化功率
maxPwDown=climbingDownRate*Pw_max #1h的最大向下变化功率
Cw_in=8000000  #购置成本 /元
c1=(12*Cw_in)/(1.3e8*60)   #功率段1运行成本
c2=(17*Cw_in)/(9.0e7*60)  #功率段2运行成本

"""
#风电参数
Pw_max=30    #额定容量 MW  
numUnit=20   #风电机组数量

W=wind_dataset.values[10:34,2]+np.random.normal(loc=0.0,scale=0.05,size=24)  #风电功率（预测）
W=Interpolation(W,43200) #三次样条插值，从24个点变为24*1800个点
W=W*numUnit  #拓展到整个风电场的功率预测
W=np.float32(W)
W=W.reshape((24,1800))   #行代表第几个小时，列代表第几个小时的第几个2s
W_min=np.zeros(24,dtype='float32') #每小时的风电最大出力预测的最小值
for t in range(24):
    W_min[t]=W[t,:].min()
eps_wind_reg=np.random.normal(loc=0.0 , scale=0.05,size=(24,1800))    #风电调频误差  均值为0标准差为0.05的正态分布
climbingUpRate=0.2  #上爬坡率
climbingDownRate=0.15 #下爬坡率
maxPwUp=climbingUpRate*Pw_max  #1h的最大向上变化功率
maxPwDown=climbingDownRate*Pw_max #1h的最大向下变化功率
Cw_in=8000000  #购置成本 /元
c1=(12*Cw_in)/(1.3e8*60)   #功率段1运行成本
c2=(17*Cw_in)/(9.0e7*60)  #功率段2运行成本
"""

#市场价格  统一用人民币表示
is_dollar=1 # 市场价格单位判断  0人名币 1 美元
exchangeRate=6.86 #美元汇率

#能量市场参数
lambda_eng=np.float32(price_dataset.values[1::1,1])*(1+is_dollar*(exchangeRate-1)) #日前能量市场价格 ￥/MWh

#调频市场参数
lambda_cap=np.float32(price_dataset.values[1::1,3])*(1+is_dollar*(exchangeRate-1)) #调频容量价格   ￥/MWh
lambda_mil=np.float32(price_dataset.values[1::1,5])*(1+is_dollar*(exchangeRate-1)) #调频里程价格   ￥/MWh
k1=5                                #k1性能分数
k2=0                                #k2性能分数
k3=0.95                              #k3性能分数
K_aver=np.float32(0.25*(2*k1+k2+k3)*0.95)             #综合性能指标分数
regD=regD_dataset.values[0:43200:1,2]   #调频信号(预测) 取自PJM市场2015-01-01

#在调频信号上叠加一个正态分布随机数作为日前预测结果，但要保证数据在(-1,1)以内
for i in range(43200):
    tmp=regD[i]+np.random.normal(loc=0.0,scale=0.05,size=1)
    while (abs(tmp)>1):
        tmp=regD[i]+np.random.normal(loc=0.0,scale=0.05,size=1)
    regD[i]=tmp
    
regD=np.float32(regD)
regD=regD.reshape(24,1800)
regD_up_max=np.zeros(24,dtype='float32') #每小时的1800个调频信号中上调频需求最大值
regD_down_max=np.zeros(24,dtype='float32') #每小时的1800个调频信号中下调频需求最大值
#每小时调频里程标幺值计算
R=np.zeros(24,dtype='float32')     #MW
for t in range(24):
    tmp=0
    regD_up_max[t]=regD[t,:].max()
    regD_down_max[t]=regD[t,:].min()    
    for i in range(1800):
        if i< 1799:
            tmp+=abs(regD[t,i+1]-regD[t,i])
    R[t]=tmp/1800
    
print("参数设定完成。")

#释放内存 节省空间
del price_dataset
del wind_dataset
del regD_dataset
gc.collect()
print("释放算例数据内存。")

#%% 建模

model=pyscipopt.Model("wind in storage-regulation system")

#添加变量
p_da,p_reg,b_cap={},{},{}
for t in range(24):
    p_da[t]=model.addVar(vtype="C",name="p_da(%s)"%t,lb=0,ub=W_min[t])
    b_cap[t]=model.addVar(vtype="C",name="b_cap(%s)"%t,lb=0)
    for k in range(1800):
        p_reg[t,k]=model.addVar(vtype="C",name="p_reg(%s,%s)"%(t,k))
        
z=model.addVar("z")  #目标函数的线性化表达

#设置目标函数
model.setObjective(z,sense="maximize")
model.addCons(
    pyscipopt.quicksum(lambda_eng[t]*p_da[t]+(lambda_cap[t]+lambda_mil[t]*R[t])*K_aver*b_cap[t] for t in range(24) )==z       
    )

#添加约束
for t in range(24):
    model.addCons(p_da[t]+b_cap[t]<=W_min[t])  #风电预留上调频容量
    model.addCons(p_da[t]-b_cap[t]>=0)   #风电预留下调频容量
    
    for k in range(1800):
        model.addCons(p_da[t]+p_reg[t,k]<=W[t,k])  #风电最大功率约束
        #model.addCons(p_reg[t,k]==b_cap[t]*regD[t,k]+eps_wind_reg[t,k]) #风电跟踪调频信号
        
model.optimize()
sol = model.getBestSol()


#%%测试

P_reg=np.zeros((24,1800),dtype='float32')
Cw=0
for t in range(24):
    for k in range(1800):
        P_reg[t,k]==sol[b_cap[t]]*regD[t,k]+eps_wind_reg[t,k]
        P_wind=P_reg[t,k]+sol[p_da[t]]
        if P_wind > 0 and P_wind < 1.4*numUnit:
            Cw+=2*c1
        elif  P_wind > 1.4*numUnit:
            Cw+=2*c2
i_eng=np.zeros(24)
i_reg=np.zeros(24)
for t in range(24):
    i_eng[t]=lambda_eng[t]*sol[p_da[t]]
    i_reg[t]=(lambda_cap[t]+lambda_mil[t]*R[t])*sol[b_cap[t]]*K_aver

I_eng=i_eng.sum()

I_reg=i_reg.sum()



print("风电参与能量-调频市场日收益："+str(model.getObjVal()-Cw)+"元")
print("能量市场收益："+str(I_eng)+"元")
print("调频市场收益："+str(I_reg)+"元")
print("风电成本："+str(Cw)+"元")

pickle.dump(i_eng,open("result/i_eng.pkl","wb"))
pickle.dump(i_reg,open("result/i_reg.pkl","wb"))

