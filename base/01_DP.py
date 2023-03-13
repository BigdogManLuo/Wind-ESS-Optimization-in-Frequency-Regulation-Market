#%%导入库
#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import gc
import threading
import win32api,win32con
from numba import jit,guvectorize,float32,int32
import pickle
from wind_interpolate import Interpolation
from FigurePlot import showPriceData,showWindData,showPriceData_English,showRegDSignal
#%% 参数设定

price_dataset=pd.read_excel("data/market price/From_IEEE2017.xlsx")   #能量、调频市场价格
wind_dataset=pd.read_excel("data/wind/west_wind_farm.xlsx")    #风电最大出力

#风电参数
Pw_max=30    #额定容量 MW  
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


#储能电池参数
Pb_max=np.int32(6)  #最大充放功率 MW
E_max=np.int32(6)  #额定容量 MWh
E_min=np.float32(0.9) #储能最小预留容量
eta=np.float32(0.95) #充放效率
N_max=np.int32(5000) #100%DOD下最大充放次数
Cb_in=np.int32(10800000)  #初始投资成本 1080万￥


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
with open("data/D-1/regD.pkl","rb") as f:
    regD=pickle.load(f)  #调频信号

#每小时调频里程标幺值计算
R=np.zeros(24,dtype='float32')     #MW
for t in range(24):
    tmp=0
    for i in range(1800):
        if i< 1799:
            tmp+=abs(regD[t,i+1]-regD[t,i])
    R[t]=tmp/1800
    
print("参数设定完成。")

#参数可视化
showPriceData_English(lambda_eng,lambda_cap,lambda_mil)
W_real=Interpolation(wind_dataset.values[10:34,2],43200)*numUnit #风电功率实际值
showWindData(W,W_real)
showRegDSignal(regD)

#释放内存 节省空间
del price_dataset
del wind_dataset
gc.collect()
print("释放算例数据内存。")

#%%MDP元素定义

Penalty=-10000000  #惩罚项

gamma=np.float32(0.98)  #学习率

#颗粒度
step_ess=0.1 #储能状态、动作颗粒度
step_wind=0.1       #风电动作颗粒度

#状态空间定义
T=np.arange(0,24,1,dtype=int)  #时间序列
E=np.arange(E_min,E_max+step_ess,step_ess,dtype='float32')  #储能能量剩余 颗粒度为0.1

#动作空间定义
B_reg=np.arange(0,round(W.max())+step_wind,step_wind,dtype='float32') #日前调频容量投标 0~Pw_max-10MW 颗粒度为0.1 (为了降低动作空间维度剪枝)
P_wda=np.arange(0,round(W.max()-4)+step_wind,step_wind,dtype='float32')#风机能量市场总出力  0~Pw_max-10MW 颗粒度为0.1
P_bda=np.arange(-Pb_max,Pb_max+step_ess,step_ess,dtype='float32') #储能能量总出力  0~Pb_max 颗粒度为0.1

#克服浮点数精度误差，避免查找失败
for e in range(E.size):
    E[e]=round(E[e],1)
for b_reg in range(B_reg.size):
    B_reg[b_reg]=round(B_reg[b_reg],1)
for p_wda in range(P_wda.size):
    P_wda[p_wda]=round(P_wda[p_wda],1)
for p_bda in range(P_bda.size):
    P_bda[p_bda]=round(P_bda[p_bda],1)

#状态值函数
V=np.zeros((len(T),len(E)),dtype='float32')   #与S同维度

#策略定义  初始为均匀策略
pi=np.ones((len(T),len(E),len(B_reg),len(P_wda),len(P_bda)),dtype='float32')*(1/(len(B_reg)*len(P_wda)*len(P_bda)))

pickle.dump(T,open('result/D-1/T.pkl','wb'))
pickle.dump(E,open('result/D-1/E.pkl','wb'))
pickle.dump(P_wda,open('result/D-1/P_wda.pkl','wb'))
pickle.dump(B_reg,open('result/D-1/B_reg.pkl','wb'))
pickle.dump(P_bda,open('result/D-1/P_bda.pkl','wb'))

print("MDP元素定义完成")

#%% 初始化即时回报矩阵 预编译加速
r=np.zeros((len(T),len(E),len(B_reg),len(P_wda),len(P_bda)),dtype='float32')
E_NEXT=np.zeros((len(T),len(E),len(B_reg),len(P_wda),len(P_bda)),dtype='float32')
@guvectorize([(int32[:],float32[:],float32[:],float32[:],float32[:],float32[:,:,:,:,:],float32[:,:,:,:,:])],
             "(t),(e),(m),(n),(k)->(t,e,m,n,k),(t,e,m,n,k)")
def initR(T,E,B_reg,P_wda,P_bda,r,E_NEXT):
    
    """
    Parameters
    ----------
    T :     状态空间   时间序列 int32 24x1 
    E :     状态空间  储能能量剩余 float32 
    B_reg : 动作空间 调频容量投标 float32
    P_wda : 动作空间 风机日前能量功率基点 float32
    P_bda : 动作空间 储能日前能量功率基点 float32
    
    Returns
    -------
    r:即时回报矩阵  TxExB_regxP_wda_P_bda
    (guvectorize 装饰器没有返回参数，但是参数r可以作为实际调用时的返回值)
    E_NEXT:在状态（t,e）下采取动作(b_reg,P_wda,P_bda)得到的下一个储能能量剩余状态e_next
    """
    #即时回报定义
    for t in range(T.size):
        for p_wda in range(P_wda.size):
            #判断风电功率是否在禁止域
            if P_wda[p_wda] > W_min[t] :
                r[t,:,:,p_wda,:]=Penalty
                continue         #剪枝
            for p_bda in range(P_bda.size):
                for b_reg in range(B_reg.size):
                    
                    #计算风电和储能预留的上下调频容量
                    Pw_rup=min(W_min[t]-P_wda[p_wda],maxPwUp)
                    Pw_rdn=min(P_wda[p_wda],maxPwDown)
                    Pb_rup=Pb_max-P_bda[p_bda]
                    Pb_rdn=Pb_max+P_bda[p_bda]
                    
                    #上调频容量约束
                    if Pw_rup+Pb_rup<B_reg[b_reg]:
                        r[t,:,b_reg,p_wda,p_bda]=Penalty
                        continue   #剪枝
                    #下调频容量约束
                    if Pw_rdn+Pb_rdn<B_reg[b_reg]:
                        r[t,:,b_reg,p_wda,p_bda]=Penalty
                        continue   #剪枝
                        
                    for e in range(E.size): 
                        #计算风电和储能的调频功率
                        P_breg=np.zeros(1800,dtype='float32')
                        cost_b=0   #储能充放成本
                        E_next=E[e]
                        for k in range(1800):
                            if regD[t,k]*B_reg[b_reg]>min(W[t,k]-P_wda[p_wda],maxPwUp):#如果上调频需求超过了风的最大上调容量
                                P_breg[k]=regD[t,k]*B_reg[b_reg]-min(W[t,k]-P_wda[p_wda],maxPwUp)+eps_wind_reg[t,k]  #储能出力来补这个差
                            elif regD[t,k]*B_reg[b_reg]<-Pw_rdn:  #如果下调频需求超过了最大下调容量
                                P_breg[k]=regD[t,k]*B_reg[b_reg]+Pw_rdn+eps_wind_reg[t,k]   #储能充电来补这个差
                            else:
                                P_breg[k]=eps_wind_reg[t,k]                            #储能补风电调频误差
                                
                            #计算储能充放成本
                            if P_bda[p_bda]+P_breg[k]>=0: #如果储能的净充放方向为放电
                                deltaE=((P_bda[p_bda]+P_breg[k])/eta)*0.0005556  #这2s的能量变化
                                cost_b+=((deltaE*Cb_in)/(2*N_max*E_max))     #这2s的放电成本
                            elif P_bda[p_bda]+P_breg[k] < 0: #如果储能的净充放方向为充电
                                deltaE=((P_bda[p_bda]+P_breg[k])*eta)*0.0005556  #这2s的能量变化
                                cost_b+=(((-deltaE)*Cb_in)/(2*N_max*E_max))#这2s的充电成本
                            #储能能量转换 2s分辨率的数值积分
                            E_next-=deltaE
                        """
                        #计算储能能量转换方程，判断下一时刻能量剩余是否越界
                        P_b=(P_breg.sum()*0.0005556)+P_bda[p_bda]  #储能第t小时净出力  =能量基点加上以一个小时内以2s为分辨率的数值积分
                        if  P_b >= 0 : #如果是放电
                            E_next=E[e]-(P_b/eta)*1
                        elif P_b <0:  #如果是充电
                            E_next=E[e]-P_b*eta*1
                        """
                        #在状态（t,e）下采取动作(b_reg,P_wda,P_bda)得到的下一个储能能量剩余状态e_next储存起来
                        #注意该E_next可能不在状态空间，需要进行round
                        E_NEXT[t,e,b_reg,p_wda,p_bda]=E_next 
                        
                        #储能电池容量约束
                        if E_next < E_min or E_next > E_max:    #如果储能容量越限了
                            r[t,e,b_reg,p_wda,p_bda]=Penalty

                        else:
                            #计算有效的即时回报
                            r_eng=lambda_eng[t]*(P_bda[p_bda]+P_wda[p_wda])#能量市场收益
                            r_reg=(lambda_cap[t]+lambda_mil[t]*R[t])*K_aver*B_reg[b_reg]*1  #调频市场收益
                            
                            #t时刻总收入=两个市场收益-储能成本
                            r[t,e,b_reg,p_wda,p_bda]=r_eng+r_reg-cost_b
       

start=time.time()
print("正在初始化即时回报矩阵...")
r,E_NEXT=initR(T,E,B_reg,P_wda,P_bda)
print("初始化即时回报矩阵完成（guvectorize装饰器加速），耗时："+str(time.time()-start)+"秒")

#写入文件保存
print("正在写入文件保存...")
pickle.dump(r,open('result/D-1/r.pkl','wb'))
pickle.dump(E_NEXT,open('result/D-1/E_NEXT.pkl','wb'))
#%% 必要时直接从文件中读取r和E_NEXT，不用花时间重复生成
with open('result/D-1/r.pkl','rb') as f:
    r=pickle.load(f)
with open('result/D-1/E_NEXT.pkl','rb') as f:
    E_NEXT=pickle.load(f)
#%% 迭代求解Bellman Optimal Equation
@jit(nopython=True)
def valueIteration(V,pi):
    newPi=pi
    newV=V
    for t in range(T.size):
        for e in range(E.size):
            maxQ=Penalty
            for b_reg in range(B_reg.size):
                for p_wda in range(P_wda.size):
                    for p_bda in range(P_bda.size):
                        #计算q(s,a) -> tmpQ
                        if r[t,e,b_reg,p_wda,p_bda]==Penalty:   #如果是forbidden action
                            tmpQ=Penalty  
                        else:
                            e_next=np.where(E==round(E_NEXT[t,e,b_reg,p_wda,p_bda],1))[0][0]  #找到在当前状态-动作下，储能的下一个状态在状态空间的索引
                            if t<23:
                                tmpQ=r[t,e,b_reg,p_wda,p_bda]+gamma*V[t+1,e_next] #计算状态-动作值
                            elif t==23:
                                tmpQ=r[t,e,b_reg,p_wda,p_bda]+gamma*V[t,e_next]
                        #选择排序法 找到最大的q(s,a) 和a*
                        if tmpQ>maxQ:
                            maxQ=tmpQ
                            b_reg_best,p_wda_best,p_bda_best=(b_reg,p_wda,p_bda)

            #pi(s)策略更新
            newPi[t,e,:,:,:]=0
            newPi[t,e,b_reg_best,p_wda_best,p_bda_best]=1
            #v(s)值函数更新
            newV[t,e]=maxQ

    return newV,newPi


#%%

lastSumV=Penalty
count=0
recordV=[]
#开始计时
start=time.time()
print("正在迭代求解Bellma最优方程....")
while(not(abs(lastSumV-V.sum())<=100)):
#for i in range(50):
    count+=1
    lastSumV=V.sum()
    V,pi=valueIteration(V,pi)
    if count==10:
        recordV.append(lastSumV)  #记录收敛过程
        count=0
#结束计时
end=time.time()

print("Bellman最优方程迭代完成！用时："+str(end-start)+"秒")

#%%
#写入文件保存
pickle.dump(V,open('result/D-1/V.pkl','wb'))
pickle.dump(pi,open('result/D-1/pi.pkl','wb'))
print("已保存状态值函数、最终策略")

#%% 必要时直接读取值函数和策略，避免重复生成
with open("result/D-1/pi.pkl",'rb') as f:
    pi=pickle.load(f)
with open("result/D-1/V.pkl",'rb') as f:
    V=pickle.load(f)
#%%运行测试

e=np.where(E==3)[0][0] #储能的初始状态  
soc_record=np.zeros(24,dtype='float32') #每个小时储能的soc状态
B_ENG=np.zeros(24,dtype='float32') #每个小时的能量投标记录
B_REG=np.zeros(24,dtype='float32') #每个小时的调频容量投标记录
P_WDA=np.zeros(24,dtype='float32') #每个小时的风电能量基点记录
P_BDA=np.zeros(24,dtype='float32') #每个小时的储能能量基点记录
P_WREG=np.zeros((24,1800),dtype='float32') #每个小时每个2s的风电调频功率记录
P_BREG=np.zeros((24,1800),dtype='float32') #每个小时每个2s的储能调频功率记录
I_ENG=np.zeros(24,dtype='float32')# 每小时能量市场收益
I_REG=np.zeros(24,dtype='float32')# 每小时调频市场收益
I_SUM=np.zeros(24,dtype='float32')# 每小时总收益
C_BES=np.zeros(24,dtype='float32') #每小时储能成本
C_WIND=np.zeros(24,dtype='float32')#每小时风电成本
P_WRUP=np.zeros(24,dtype='float32')  #风电预留上调频量


sumR=0 #累计收益

for t in range(T.size):
    
    #记录每小时储能状态
    soc_record[t]=E[e]/E_max
    
    #在最终策略pi中找到最优动作
    b_reg,p_wda,p_bda=np.unravel_index(np.argmax(pi[t,e,:,:,:]),pi[t,e,:,:,:].shape)
    
    
    #记录该动作
    B_REG[t]=B_reg[b_reg]
    P_WDA[t]=P_wda[p_wda]
    P_BDA[t]=P_bda[p_bda]
    
    #计算风电和储能预留的上下调频容量
    Pw_rup=min(W_min[t]-P_wda[p_wda],maxPwUp)
    Pw_rdn=min(P_wda[p_wda],maxPwDown)
    Pb_rup=Pb_max-P_bda[p_bda]
    Pb_rdn=Pb_max+P_bda[p_bda]
    P_WRUP[t]=Pw_rup
    
    #记录该动作下的风储调频功率
    E_next=E[e]
    for k in range(1800):
        
        if regD[t,k]*B_reg[b_reg]>=min(W[t,k]-P_wda[p_wda],maxPwUp):#如果上调频需求超过了风的最大上调容量
            P_WREG[t,k]=min(W[t,k]-P_wda[p_wda],maxPwUp)
            P_BREG[t,k]=regD[t,k]*B_reg[b_reg]-Pw_rup+eps_wind_reg[t,k]
            
        elif regD[t,k]*B_reg[b_reg]<=-Pw_rdn:  #如果下调频需求超过了风电的最大下调容量
            P_WREG[t,k]=-Pw_rdn               
            P_BREG[t,k]=regD[t,k]*B_reg[b_reg]+Pw_rdn+eps_wind_reg[t,k]  
        else:
            P_WREG[t,k]=regD[t,k]*B_reg[b_reg]       #跟踪调频需求信号
            P_BREG[t,k]=eps_wind_reg[t,k]                         #储能不动作
    
        #计算储能充放成本
        if P_bda[p_bda]+P_BREG[t,k]>=0: #如果储能的净充放方向为放电
            deltaE=((P_bda[p_bda]+P_BREG[t,k])/eta)*0.0005556  #这2s的能量变化
            C_BES[t]+=((deltaE*Cb_in)/(2*N_max*E_max))     #这2s的放电成本
        elif P_bda[p_bda]+P_BREG[t,k] < 0: #如果储能的净充放方向为充电
            deltaE=((P_bda[p_bda]+P_BREG[t,k])*eta)*0.0005556  #这2s的能量变化
            C_BES[t]+=(((-deltaE)*Cb_in)/(2*N_max*E_max))#这2s的充电成本
        
        
        #计算风电成本
        if P_wda[p_wda]+P_WREG[t,k]<=37.33:
            C_WIND[t]+=2*c1
        elif P_wda[p_wda]+P_WREG[t,k]>37.33:
            C_WIND[t]+=2*c2
    
    
    #记录收益
    B_ENG[t]=P_WDA[t]+P_BDA[t]      
    I_ENG[t]=lambda_eng[t]*B_ENG[t] #能量市场收益
    I_REG[t]=(lambda_cap[t]+lambda_mil[t]*R[t])*K_aver*B_REG[t] #调频市场收益
    I_SUM[t]=r[t,e,b_reg,p_wda,p_bda]-C_WIND[t]#总收益
    
    #状态转换
    E_next=E_NEXT[t,e,b_reg,p_wda,p_bda]
    E_next=round(E_next,1)
    e_next=np.where(E==E_next)[0][0]
    e=e_next
    
    
#数据整理
P_WREG_SUM=np.array([P_WREG[i,:].sum() for i in range(P_WREG.shape[0])])*0.0005556  #每小时风电总调频功率
P_BREG_SUM=np.array([P_BREG[i,:].sum() for i in range(P_BREG.shape[0])])*0.0005556  #每小时储能总调频功率

#收益数据统计
print("能量市场收益："+str(I_ENG.sum())+"元")
print("调频市场收益："+str(I_REG.sum())+"元")
print("储能成本："+str(C_BES.sum())+"元")
print("风电成本："+str(C_WIND.sum())+"元")
print("总收益："+str(I_SUM.sum())+"元")
#%%保存日前优化数据
pickle.dump(P_WDA,open('result/D-1/P_WDA.pkl','wb'))
pickle.dump(P_BDA,open('result/D-1/P_BDA.pkl','wb'))
pickle.dump(B_REG,open('result/D-1/B_REG.pkl','wb'))

#%%可视化 求解结果

plt.rcParams['font.sans-serif']=['SongNTR'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#绘图
plt.bar(np.arange(24),P_WDA,width=0.6,label="风电能量基点",color="#2878b5")
plt.bar(np.arange(24),P_WRUP,width=0.6,label="风电调频预留",color="#9ac9db",bottom=P_WDA)
plt.plot(np.arange(24),P_BDA+P_BREG_SUM,label="储能运行功率",color="#b8e994",linewidth=2,marker="x")
plt.plot(np.arange(0,24),W_min,color="#c82423",label="风电功率预测",linewidth=2,marker=".")
plt.plot(np.arange(24),B_ENG,color="#f8ac8c",label="能量市场投标",linewidth=2,marker=".")
plt.plot(np.arange(24),B_REG,color="#ff8884",label="调频容量投标",linewidth=2,marker=".")

#去掉上框线和右框线
ax=plt.gca()  #gca:get current axis得到当前轴
#设置图片的右边框和上边框为不显示
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

#定义图例的字体
plt.legend(frameon=False)
#添加x标签，y标签
plt.xlabel("时间/h",size=16,font='SongNTR')
plt.ylabel("功率/MW",size=16,font='SongNTR')
plt.xticks(np.arange(0,24,2),family="Times New Roman")
plt.yticks(np.arange(0,20,2),family="Times New Roman")

plt.grid()
plt.savefig("figure/D-1/result.png",dpi=500)

plt.show()

#%%可视化 收益随时间变化曲线
plt.plot(np.arange(24),I_ENG,label="能量市场收益",marker=".")
plt.plot(np.arange(24),I_REG,label="调频市场收益",marker=".")
plt.plot(np.arange(24),C_BES,label="储能运行成本",marker=".")
plt.grid()
plt.xlabel("时间/h",size=15,font='SongNTR')
plt.ylabel("市场收益/￥",size=15,font='SongNTR')
plt.xticks(np.arange(0,24,2),family="Times New Roman")
plt.yticks(np.arange(0,40000,5000),family="Times New Roman")
plt.legend(frameon=False)
plt.savefig("figure/D-1/income.png",dpi=500)
#%% 可视化 储能soc状态
plt.plot(np.arange(24),soc_record,marker=".")
plt.xlabel("时间/h",size=15,font='SongNTR')
plt.ylabel("荷电状态",size=15,font='SongNTR')
plt.xticks(np.arange(0,24,2),family="Times New Roman")
plt.yticks(np.arange(0,1,0.1),family="Times New Roman")
plt.grid()
plt.savefig("figure/D-1/soc.png",dpi=500)
#%%单个断面的运行状态
for t in range(23,24):
    print("正在生成第"+str(t)+"小时断面风储运行状态图("+str(t)+"/23)")
    plt.figure()
    P_WDA_draw=np.ones(1800,dtype='float32')*P_WDA[t]
    plt.plot(np.arange(1800),W[t,:],label="风电最大出力预测",color="#fdcb6e")
    plt.plot(np.arange(1800),B_REG[t]*regD[t,:]+P_WDA[t],label="实际调频需求",color="#e17055")
    plt.bar(np.arange(1800),P_WDA_draw,width=1,label="风电能量基点",color="#dfe6e9")
    plt.bar(np.arange(1800),P_WREG[t,:],width=1,bottom=P_WDA_draw,label="风电调频出力",color="#74b9ff")
    plt.bar(np.arange(1800),P_BREG[t,:],width=1,bottom=P_WDA_draw+P_WREG[t,:],label="储能调频出力",color="#00b894") 
    plt.title("第"+str(t)+"个小时风储运行状态(MW)")
    plt.xlabel("时间/s",size=15,font='SongNTR')
    plt.xticks(np.arange(0,1800+150,150),family="Times New Roman")
    plt.yticks(np.arange(0,35,5),family="Times New Roman")
    plt.grid()
    plt.legend(frameon=False)
    path="figure/D-1/Cross-section"+str(t)
    plt.savefig(path,dpi=500)
#%% 值函数曲面(收敛最终结果)
x,y=np.meshgrid(E,T)
# 绘制图片
fig = plt.figure("3D Surface", facecolor="lightgray",figsize=(10,7.5))

plt.title("State Value Surface(end of iteration)", fontsize=18,family="Times New Roman")

# 设置为3D图片类型
#ax3d = Axes3D(fig)
ax3d = fig.add_subplot(projection="3d")    # 同样可以实现

ax3d.plot_surface(x,y, V,cmap='inferno')
ax3d.set_xlabel("Energy Remain of ESS (MWh)")
ax3d.set_ylabel("Time(hour)")
ax3d.set_zlabel("state value")
plt.tick_params(labelsize=10)
plt.savefig("figure/stateValue1.png",dpi=500)
plt.show()

#%%值函数曲面 初始值迭代
x,y=np.meshgrid(E,T)
# 绘制图片
fig = plt.figure("3D Surface", facecolor="lightgray",figsize=(10,7.5))

plt.title("State Value Surface(beginning of iteration)", fontsize=18,family="Times New Roman")

# 设置为3D图片类型
#ax3d = Axes3D(fig)
ax3d = fig.add_subplot(projection="3d")    # 同样可以实现

ax3d.plot_surface(x,y, V,cmap='inferno')
ax3d.set_xlabel("Energy Remain of ESS (MWh)")
ax3d.set_ylabel("Time(hour)")
ax3d.set_zlabel("state value")
plt.tick_params(labelsize=10)
plt.savefig("figure/stateValue0.png",dpi=500)
plt.show()
#%% 可视化  迭代过程

plt.plot(recordV,marker='.')
plt.xlabel("迭代次数/10次",size=13,font='SongNTR')
plt.ylabel("V(s)",size=13,font='SongNTR')
plt.title("值函数迭代收敛过程")
plt.grid()
plt.savefig("figure/state iteration process.png",dpi=500)
plt.show()


#%%内存占用统计
import sys
print(str(((sys.getsizeof(pi)/1024)/1024)/1024)+"GB")


#%%弹窗函数 提示程序正在运行
def msgboxFunc(info):
    win32api.MessageBox(0,info,"注意",win32con.MB_OK)

#开启一个新的线程
info="当前程序正在运行，请勿关闭,谢谢！"
th=threading.Thread(target=msgboxFunc,args=(info,))
th.start()


