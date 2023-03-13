import matplotlib.pyplot as plt
import pickle 
import numpy as np

x=np.arange(3)
labels=["Wind-ESS in \n Energy-Reg Market","Wind Power \n in Energy-Reg Market","Wind-ESS \n in Energy Market"]

with open("result/wind_ESS_Eng_Reg/I_ENG.pkl","rb") as f:
    y1_eng=pickle.load(f).sum()
with open("result/wind_ESS_Eng_Reg/I_REG.pkl","rb") as f:
    y1_reg=pickle.load(f).sum()

y1_bes=2008.10
    
with open("result/wind_Eng_Reg/i_eng.pkl","rb") as f:
    y2_eng=pickle.load(f).sum()
with open("result/wind_Eng_Reg/i_reg.pkl","rb") as f:
    y2_reg=pickle.load(f).sum()


y3_eng=85855.18+13478.79

y_eng=[y1_eng,y2_eng,y3_eng]
y_reg=[y1_reg,y2_reg,0]
y_bes=[y1_bes,0,0]

plt.grid(axis="y",zorder=0)
plt.bar(x,y_eng,zorder=10,label="Engergy",width=0.4,color="#2878b5")
plt.bar(x,y_reg,bottom=y_eng,zorder=10,label="Regulation",width=0.4,color="#9AC9DB")
plt.xticks(x,labels,family="songNTR")
plt.ylabel("Revenue(￥)",family="songNTR",size=12)
font = {'family' : 'Times New Roman',
'weight':'normal',
'size': 10,
}
plt.legend(prop=font,ncol=1,frameon=False)

plt.savefig("figure/Revenue_Comparision.png",dpi=1000)
    
    
