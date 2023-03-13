#%%
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wind_interpolate import Interpolation
from numba import njit
import time
from pyscipopt import Model,quicksum
import math
#%%
class DataLoader:
    
    def __init__(self,DAResultFolder,RTDataFolder):
        self.RTDataFolder=RTDataFolder
        self.DAResultFolder=DAResultFolder
    
    def readWind(self,fileName,U,interval):
        """
        函数功能：从文件中读取实际运行日风电功率
        输入参数： 
            fileName RTDataFolder路径下的风电数据文件名
            U：风电机组总数
        返回值：
            W_rt 24x1800 风电场整体功率
            W_perUnit  24*(60/interval)x interval*30 单台风机功率
            W_rt_min   24x1 每个断面风机最大功率最小值
        """ 
        wind_dataset=pd.read_excel(self.RTDataFolder+fileName)
        W=wind_dataset.values[10:34,2]
        W=Interpolation(W,43200) #三次样条插值，从24个点变为24*1800个点
        W_perUnit=W.reshape(( int(24*(60/interval)),int(interval*30)))
        W_rt=W*U  #拓展到整个风电场的功率预测
        W_rt=W_rt.reshape((24,1800))   #行代表第几个小时，列代表第几个小时的第几个2s
        W_rt_min=np.zeros(24,dtype='float32') #每小时的风电最大出力预测的最小值
        for t in range(24):
            W_rt_min[t]=W_rt[t,:].min()
        return W_perUnit,W_rt,W_rt_min
    
    
    def readRegD(self,fileName):
        """
        函数功能：从文件中读取实际运行日调频信号
        输入参数： fileName RTDataFolder路径下的调频信号数据文件名
        返回值： regD 24x1800
        """
        regD_dataset=pd.read_excel(self.RTDataFolder+fileName) #调频信号
        regD=regD_dataset.values[0:43200:1,2]   #取自PJM市场2015-01-01
        regD=regD.reshape(24,1800)
        regD=np.float32(regD)
        return regD
    
    def readWindRegEps(self,fileName):
        """
        #读取风电调频误差
        返回值：eps_wind_reg 24x1800
        """
        with open(self.RTDataFolder+fileName,"rb") as f:
            eps_wind_reg=pickle.load(f)   #风电调频误差  均值为0标准差为0.05的正态分布
        eps_wind_reg=eps_wind_reg.reshape(24,1800)
        
        return eps_wind_reg
        
        
    def readV(self):
        with open(self.DAResultFolder+"V.pkl","rb") as f:
            V=pickle.load(f)
            return V
    
    def readStateSpace(self):
        with open(self.DAResultFolder+"T.pkl","rb") as f:
            T=pickle.load(f)
        with open(self.DAResultFolder+"E.pkl","rb") as f:
            E=pickle.load(f)
            return T,E
    def readActionSpace(self):
        with open(self.DAResultFolder+"P_wda.pkl","rb") as f:
            P_wda=pickle.load(f)
        with open(self.DAResultFolder+"B_reg.pkl","rb") as f:
            B_reg=pickle.load(f)
        with open(self.DAResultFolder+"P_bda.pkl","rb") as f:
            P_bda=pickle.load(f)
        return B_reg,P_wda,P_bda
        
class WINDUC:
    
    def __init__(self,UCResultFolder,t,numUnit):
        self.UCResultFolder=UCResultFolder
        self.t=t
        self.numUnit=numUnit
    
    def readSolU(self):
        """
        函数功能：从文件中读取上一个断面风电机组的起停状态
        输入参数： filePath 文件所在路径
        返回值： [u1,u2,...ui] 列表
        """
        with open(self.UCResultFolder+str(self.t-1)+".txt") as f:
            tmp1=f.read()
            tmp2=tmp1.split("\n")
            tmp3=[float(tmp2[i]) for i in range(len(tmp2))]
        return tmp3

    def windUC(self,P_all,interval):
        """
        风机组合优化
    
        Parameters
        ----------
        t:当前断面时刻
        P_all : 风电在当前断面断面每2s分辨率的实时出力
        interval:多久一个断面 60的整数倍
        Returns
        -------
        当前断面的风机成本
    
        """
        t=self.t
        U=self.numUnit
        model=Model("UC in halfHour"+str(t))
        
        if t==0: #初始状态为全开机
            u_lastTime=[1 for i in range(U)] #上一个时刻机组的起停状态
        else:
            #获取上一个时刻的起停状态
            u_lastTime=self.readSolU()
        
        #添加变量
        print("正在添加第"+str(t)+"/"+str(24*(60/interval)-1)+"个风机组合优化模型变量")
        x,y,u,p={},{},{},{}
        for i in range(U):
            u[i]=model.addVar(vtype="B",name="u(%s)"%i)
            for k in range(int(interval*30)):
                x[i,k]=model.addVar(vtype="B",name="x(%s,%s)"%(i,k))
                y[i,k]=model.addVar(vtype="B",name="y(%s,%s)"%(i,k))
                p[i,k]=model.addVar(vtype="C",name="p(%s,%s)"%(i,k),lb=0,ub=1.5)
            
        z=model.addVar("z")  #目标函数的线性化表达
        
        #设置目标函数
        model.setObjective(z,sense="minimize")
        model.addCons(
            quicksum(quicksum(u[i]*(x[i,k]*a+y[i,k]*b)*2 for k in range(int(interval*30)))+u[i]*(1-u_lastTime[i])*c + u_lastTime[i]*(1-u[i])*d for i in range(U))==z       
            )
    
        #约束
        print("正在添加第"+str(t)+"/"+str(24*(60/interval)-1)+"个风机组合优化模型约束")
        for i in range(U):
            for k in range(int(interval*30)):
                model.addCons(x[i,k]+y[i,k]==1)   #状态唯一性
                model.addCons(x[i,k]*(p[i,k]-1.4)<=0)  #判断风机是否在功率段1
                model.addCons(y[i,k]*(p[i,k]-1.4)>=0) #判断风机是否在功率段2
                model.addCons(p[i,k]<=W_perUnit[t,k]) #风机总功率小于预测最大出力
                #当ui=0的时候将风机功率拉到0 
                model.addCons(p[i,k]<=0+u[i]*1.5)
                model.addCons(p[i,k]>=0-u[i]*1.5)
        
        for k in range(int(interval*30)):        
            model.addCons(quicksum(p[i,k] for i in range(U))==P_all[k]) #风机组合
        
        #模型求解
        print("正在求解第"+str(t)+"/"+str(24*(60/interval)-1)+"个风机组合优化模型")
        start=time.time()
        model.optimize()
        sol = model.getBestSol()
        print("第"+str(t)+"/"+str(24*(60/interval)-1)+"个风机组合优化模型求解完成，用时"+str(-start+time.time())+"秒")
        
        #导出求解结果
        with open(self.UCResultFolder+str(t)+".txt","w") as f:
            for i in range(U):
                f.write(str(sol[u[i]]))
                if i<U-1:
                    f.write("\n")
        
        #返回当前断面的风机成本
        return model.getObjVal()

#实时优化算法
@njit
def rollOut(t,e):
    
    immediateR=np.float32(0)
    bestImmediateR=np.float32(0)
    bestsumR=np.float32(0)
    bestE_next=E[e]

    cumR=np.float32(0)
    sumR=np.float32(0)
    for p_wda in range(P_wda.size):  
        for b_reg in range(B_reg.size):
            for p_bda in range(P_bda.size):
            
                #风电功率基点约束
                if P_wda[p_wda] > W_rt_min[t]:
                    immediateR=-1000000
                    cumR=--1000000
                else:
                    #计算风电和储能预留的上下调频容量
                    Pw_rup=min(W_rt_min[t]-P_wda[p_wda],maxPwUp)
                    Pw_rdn=min(P_wda[p_wda],maxPwDown)
                    Pb_rup=Pb_max-P_bda[p_bda]
                    Pb_rdn=Pb_max+P_bda[p_bda]
                
                    #上下调频容量约束
                    if Pw_rup+Pb_rup < B_reg[b_reg] or Pw_rdn+Pb_rdn < B_reg[b_reg]:
                        immediateR=-1000000
                        cumR=-1000000
                    else:
                        #计算储能的调频功率和充放成本
                        cost_b=0
                        P_breg=0.0
                        E_next=E[e]
                        for k in range(1800):
                            if regD_rt[t,k]*B_reg[b_reg]>min(W_rt[t,k]-P_wda[p_wda],maxPwUp):#如果上调频需求超过了风的最大上调容量
                                P_breg=regD_rt[t,k]*B_reg[b_reg]-min(W_rt[t,k]-P_wda[p_wda],maxPwUp)+eps_wind_reg[t,k]  #储能出力来补这个差
                            elif regD_rt[t,k]*B_reg[b_reg]<-Pw_rdn:  #如果下调频需求超过了最大下调容量
                                P_breg=regD_rt[t,k]*B_reg[b_reg]+Pw_rdn+eps_wind_reg[t,k]   #储能充电来补这个差
                            else:
                                P_breg=eps_wind_reg[t,k]                            #储能补风电调频误差
                                
                            #计算储能充放成本
                            P_b=P_bda[p_bda]+P_breg
                            if P_b>=0: #如果储能的净充放方向为放电
                                deltaE=((P_b)/eta)*0.0005556  #这2s的能量变化
                                cost_b+=((deltaE*Cb_in)/(2*N_max*E_max))     #这2s的放电成本
                            elif P_b < 0: #如果储能的净充放方向为充电
                                deltaE=((P_b)*eta)*0.0005556  #这2s的能量变化
                                cost_b+=(((-deltaE)*Cb_in)/(2*N_max*E_max))#这2s的充电成本
                            
                            #储能能量转换 2s分辨率的数值积分
                            E_next=E_next-deltaE
                        
                        #储能电池容量约束
                        if E_next < E_min or E_next > E_max:    #如果储能容量越限了
                            immediateR=-1000000
                            cumR=-1000000
                        else:  
                            #计算当前动作下的即时回报
                            r_eng=lambda_eng[t]*(P_bda[p_bda]+P_wda[p_wda])#能量市场收益
                            r_reg=(lambda_cap[t]+lambda_mil[t]*R[t])*K_aver*B_reg[b_reg]  #调频市场收益
                            immediateR=r_eng+r_reg-cost_b
                            
                            #搜索状态空间中E_next对应哪一个状态
                            E_next=round(E_next,1)
                            for m in range(E.size):
                                if abs(E_next - E[m])<=0.01:
                                    e_next=m
                                    
                            #时间过渡
                            if t==23:
                                t_next=t
                            elif t<23:
                                t_next==t+1
                            
                            #计算采取当前动作过渡到下一个状态之后，按pi进行决策的累积收益
                            cumR=V[t_next,e_next]
    
                #计算总收益
                sumR=immediateR+cumR
                
                #选择寻优法
                if sumR > bestsumR:
                    
                    bestsumR=sumR
                    
                    bestb_reg=b_reg
                    bestp_wda=p_wda
                    bestp_bda=p_bda
                    
                    bestImmediateR=immediateR
                    bestE_next=E_next
            
    
    return bestb_reg,bestp_wda,bestp_bda,bestImmediateR,bestE_next
      

      
#%% 参数设定
DAResultFolder="result/D-1/"
RTDataFolder="data/D/"
dataloader=DataLoader(DAResultFolder,RTDataFolder)

#风电参数
interval=30 #机组组合间隔
Pw_max=30    #额定容量 MW  
numUnit=20   #风电机组数量
eps_wind_reg=dataloader.readWindRegEps("eps_wind_reg.pkl")
W_perUnit,W_rt,W_rt_min=dataloader.readWind("west_wind_farm.xlsx", U=numUnit,interval=interval)
climbingUpRate=0.2  #上爬坡率
climbingDownRate=0.15 #下爬坡率
maxPwUp=climbingUpRate*Pw_max  #1h的最大向上变化功率
maxPwDown=climbingDownRate*Pw_max #1h的最大向下变化功率
Cw_in=8000000  #购置成本 /元
Cw_in_perUnit=Cw_in
a=(12*Cw_in_perUnit)/(1.3e8*60)   #功率段1运行成本
b=(17*Cw_in_perUnit)/(9.0e7*60)  #功率段2运行成本
c=(12*Cw_in_perUnit)/(1.3e8)    #启动成本
d=(2.5*Cw_in_perUnit)/(1.3e8)   #停机成本


#储能电池参数
Pb_max=6  #最大充放功率 MW
E_max=6.0  #额定容量 MWh
E_min=0.9
eta=0.95 #充放效率
N_max=5000 #100%DOD下最大充放次数
Cb_in=10800000  #初始投资成本 1080万￥


#实际调频信号
regD_rt=dataloader.readRegD("regD.xlsx")

#市场参数
is_dollar=1 # 市场价格单位判断  0人名币 1 美元
exchangeRate=6.86 #美元汇率
price_dataset=pd.read_excel("data/market price/From_IEEE2017.xlsx")   #能量、调频市场价格
lambda_eng=np.float32(price_dataset.values[1::1,1])*(1+is_dollar*(exchangeRate-1)) #日前能量市场价格 ￥/MWh

#调频市场参数
lambda_cap=np.float32(price_dataset.values[1::1,3])*(1+is_dollar*(exchangeRate-1)) #调频容量价格   ￥/MWh
lambda_mil=np.float32(price_dataset.values[1::1,5])*(1+is_dollar*(exchangeRate-1)) #调频里程价格   ￥/MWh
k1=5                                #k1性能分数
k2=0                                #k2性能分数
k3=0.95                              #k3性能分数
K_aver=np.float32(0.25*(2*k1+k2+k3)*0.95)             #综合性能指标分数
#每小时调频里程标幺值计算
R=np.zeros(24,dtype='float32')     #MW
for t in range(24):
    tmp=0
    for i in range(1800):
        if i< 1799:
            tmp+=abs(regD_rt[t,i+1]-regD_rt[t,i])
    R[t]=tmp/1800
#%%MDP元素定义

Penalty=-10000000  #惩罚项

T,E=dataloader.readStateSpace()
B_reg,P_wda,P_bda=dataloader.readActionSpace()
V=dataloader.readV()

#制作一个最优策略表，用于查询在状态t,e下的最优策略
"""
bestPi=[[i for j in range(E.size)]for i in range(T.size)]
for t in range(T.size):
    for e in range(E.size):
       bestPi[t][e]=np.unravel_index(np.argmax(pi[t,e,:,:,:]),pi[t,e,:,:,:].shape)
"""

#%% 运行测试
  
         
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
    
    #计时
    start=time.time()
    
    #记录每小时储能状态
    soc_record[t]=E[e]/E_max
    
    
    #rollout算法做滚动优化
    b_reg,p_wda,p_bda,immediateR,E_next=tmp=rollOut(t,e)
    
    #显示算法时间
    print("用时: {} s".format(time.time()-start))
    
    #记录该动作
    B_REG[t]=B_reg[b_reg]
    P_WDA[t]=P_wda[p_wda]
    P_BDA[t]=P_bda[p_bda]
    
    #记录预留上调频容量
    P_WRUP[t]=W_rt_min[t]-P_WDA[t]
    
    #记录风电调频功率
    for k in range(1800):
        Pw_rup=min(W_rt[t,k]-P_wda[p_wda],maxPwUp)
        Pw_rdn=min(P_wda[p_wda],maxPwDown)
        
        if regD_rt[t,k]*B_reg[b_reg]>=Pw_rup:#如果上调频需求超过了风的最大上调容量
            P_WREG[t,k]=Pw_rup
            P_BREG[t,k]=regD_rt[t,k]*B_reg[b_reg]-Pw_rup+eps_wind_reg[t,k]
        
        elif regD_rt[t,k]*B_reg[b_reg]<=-Pw_rdn:  #如果下调频需求超过了风电的最大下调容量
            P_WREG[t,k]=-Pw_rdn
            P_BREG[t,k]=regD_rt[t,k]*B_reg[b_reg]+Pw_rdn+eps_wind_reg[t,k]           
        else:
            P_WREG[t,k]=regD_rt[t,k]*B_reg[b_reg]       #跟踪调频需求信号
            P_BREG[t,k]=eps_wind_reg[t,k] 
    
    #记录收益
    B_ENG[t]=P_WDA[t]+P_BDA[t]      
    I_ENG[t]=lambda_eng[t]*B_ENG[t] #能量市场收益
    I_REG[t]=(lambda_cap[t]+lambda_mil[t]*R[t])*K_aver*B_REG[t] #调频市场收益
    I_SUM[t]=immediateR#总收益
    
    #状态转换
    E_next=round(E_next,1)
    e_next=np.where(E==E_next)[0][0]
    e=e_next
    

#保存收益情况
pickle.dump(I_ENG,open("result/D/I_ENG.pkl","wb"))
pickle.dump(I_REG,open("result/D/I_REG.pkl","wb"))
pickle.dump(C_BES,open("result/D/C_BES.pkl","wb"))

#%%可视化 求解结果

plt.rcParams['font.sans-serif']=['SongNTR'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#绘图
plt.bar(np.arange(24),P_WDA,width=0.6,label="Wind Energy Basepoint",color="#2878b5")
plt.bar(np.arange(24),P_WRUP,width=0.6,label="Regulation Capacity Reserved",color="#9ac9db",bottom=P_WDA)
plt.plot(np.arange(24),P_BDA,label="ESS Energy Basepoint ",color="#b8e994",linewidth=2,marker=".")
plt.plot(np.arange(0,24),W_rt_min,color="#c82423",label="Wind Power Forecast",linewidth=2,marker=".")
plt.plot(np.arange(24),B_ENG,color="#f8ac8c",label="Energy Market Bidding",linewidth=2,marker=".")
plt.plot(np.arange(24),B_REG,color="#ff8884",label="Regulation Market Bidding",linewidth=2,marker=".")

#去掉上框线和右框线
ax=plt.gca()  #gca:get current axis得到当前轴
#设置图片的右边框和上边框为不显示
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

#定义图例的字体
plt.legend(loc="upper right",ncol=2,frameon=False)
#添加x标签，y标签
plt.xlabel("Time/h",size=16,font='SongNTR')
plt.ylabel("Power/MW",size=16,font='SongNTR')
plt.xticks(np.arange(0,24,2),family="Times New Roman")
plt.yticks(np.arange(0,20,2),family="Times New Roman")

plt.grid()
plt.savefig("figure/D/result.png",dpi=500)

plt.show()

#%%单个断面的运行状态
P_WREG=P_WREG.reshape(24,1800)
P_BREG=P_BREG.reshape(24,1800)
for t in range(2):
    print("正在生成第"+str(t)+"小时断面风储运行状态图("+str(t)+"/23)")
    plt.figure()
    P_WDA_draw=np.ones(1800,dtype='float32')*P_WDA[t]
    plt.plot(np.arange(1800),W_rt[t,:],label="Maximum Power Output",color="#fdcb6e")
    plt.plot(np.arange(1800),B_REG[t]*regD_rt[t,:]+P_WDA[t],label="Actual Regulation Demand",color="#e17055")
    plt.bar(np.arange(1800),P_WDA_draw,width=1,label="Wind Energy Basepoint",color="#dfe6e9")
    plt.bar(np.arange(1800),P_WREG[t,:],width=1,bottom=P_WDA_draw,label="Wind Regulation Power",color="#74b9ff")
    plt.bar(np.arange(1800),P_BREG[t,:],width=1,bottom=P_WDA_draw+P_WREG[t,:],label="ESS Regulation Power",color="#00b894") 
    #plt.title("第"+str(t)+"个小时风储运行状态(MW)")
    plt.xlabel("Time/s",size=15,font='SongNTR')
    plt.ylabel("Power/MW",size=15)
    plt.xticks(np.arange(0,1800+150,150),family="Times New Roman")
    plt.yticks(np.arange(0,35,5),family="Times New Roman")
    plt.grid()
    plt.legend(frameon=False)
    path="figure/D/Cross-section"+str(t)
    plt.savefig(path,dpi=500)
#%%风机组合优化
UCResultFolder="result/D/windUC/windUC_hour"
P_WREG=P_WREG.reshape(int((60/interval)*24),int(interval*30))
Cw=0
for j in range(int((60/interval)*24)):
    windUCOptimizer=WINDUC(UCResultFolder,j,numUnit)
    P_all=P_WDA[math.floor(j/3)]+P_WREG[j,:]   #计算风电2s分辨率下的总功率
    Cw+=windUCOptimizer.windUC(P_all,interval)       #风电机组组合
    
#%%机组组合结果可视化
UCResultFolder="result/D/windUC/windUC_hour"
X=np.ones((numUnit,72)) #行为机组编号，列为每个小时
X[1,2]=0
'''
for t in range(72):
    with open(UCResultFolder+str(t)+".txt") as f:
        tmp1=f.read()
        tmp2=tmp1.split("\n")
        tmp3=[float(tmp2[i]) for i in range(len(tmp2))]
    X[:,t]=tmp3
'''    
   
plt.imshow(X,cmap="gray",origin="upper")
plt.grid() 
plt.xlabel("Time/20min",size=12,font='SongNTR')
plt.ylabel("Unit Number",size=12,font='SongNTR')
plt.xticks(np.arange(0,72+1,1),font='SongNTR')
plt.yticks(np.arange(0,numUnit+1,2),font='SongNTR')

plt.savefig("figure/windUC.png",dpi=500)    

#%%收益数据统计

print("能量市场收益："+str(I_ENG.sum())+"元")
print("调频市场收益："+str(I_REG.sum())+"元")
print("储能成本："+str(-(I_SUM.sum()-I_REG.sum()-I_ENG.sum()))+"元")
print("风电成本"+str(Cw)+"元")
print("总收益："+str(I_SUM.sum()-Cw)+"元")
    
    
    
    