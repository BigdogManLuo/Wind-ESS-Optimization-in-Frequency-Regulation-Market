#绘图 数据可视化，使得主程序更简洁
import matplotlib.pyplot as plt
import numpy as np


def showPriceData(lambda_eng,lambda_cap,lambda_mil):
    
    plt.rcParams['font.sans-serif']=['SongNTR'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])

    ax.step(np.arange(0,24),lambda_eng,color="#2e86de",label="能量市场价格")
    ax.step(np.arange(0,24),lambda_cap,color="#d63031",label="调频容量价格")
    ax.step(np.arange(0,24),lambda_mil,color="#ff9f43",label="调频里程价格")
    ax.grid()
    ax.set_xlabel("时间/h",size=15,font='SongNTR')
    ax.set_ylabel("电价(￥/MWh)",size=15,font='SongNTR')
    ax.set_xticks(np.arange(0,24,1),font='SongNTR')
    ax.set_yticks(np.arange(0,800,100),font='SongNTR')
    ax.legend(frameon=False)
    plt.savefig("figure/D-1/price_data.png",dpi=1000,bbox_inches = 'tight')
    
def showPriceData_English(lambda_eng,lambda_cap,lambda_mil):
        
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])

    ax.step(np.arange(0,24),lambda_eng,color="#2e86de",label="Energy Market")
    ax.step(np.arange(0,24),lambda_cap,color="#d63031",label="Regulation Capacity")
    ax.step(np.arange(0,24),lambda_mil,color="#ff9f43",label="Regulation Performance")
    ax.grid()
    ax.set_xlabel("Time/h",size=15,font='SongNTR')
    ax.set_ylabel("Price(￥/MWh)",size=15,font='SongNTR')
    ax.set_xticks(np.arange(0,24,1),font='SongNTR')
    ax.set_yticks(np.arange(0,800,100),font='SongNTR')
    ax.legend(frameon=False)
    plt.savefig("figure/D-1/price_data.png",dpi=1000,bbox_inches = 'tight')

def showWindData(pred,real):
    
    pred=pred.reshape(43200)
    
    plt.rcParams['font.sans-serif']=['SongNTR'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    
    x=np.arange(0,24,(1/1800))
    ax.plot(x,pred,label="预测值")
    ax.plot(x,real,label="实际值")
    ax.grid()
    ax.set_xlabel("时间/h",size=15,font='SongNTR')
    ax.set_ylabel("风电功率(MW)",size=15,font='SongNTR')
    ax.set_xticks(np.arange(0,24,2),font='SongNTR')
    ax.set_yticks(np.arange(0,24,2),font='SongNTR')
    ax.legend(frameon=False)
    plt.savefig("figure/D-1/wind_data.png",dpi=1000,bbox_inches = 'tight')
    

def showRegDSignal(regD):
    plt.rcParams['font.sans-serif']=['SongNTR'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    
    x=np.arange(0,24,(1/1800))
    regD=regD.reshape(24*1800) #拉平
    
    ax.plot(x,regD)
    ax.set_xlabel("时间/h",size=15,font='SongNTR')
    ax.set_ylabel("调频信号",size=15,font='SongNTR')
    ax.set_xticks(np.arange(0,28,4),font='SongNTR')
    ax.set_yticks(np.arange(-1,1.25,0.25),font='SongNTR')
    plt.savefig("figure/D-1/regD.png",dpi=1000,bbox_inches = 'tight')
    
    
    
    
    
    
    
    
    
    
    
    
    
    