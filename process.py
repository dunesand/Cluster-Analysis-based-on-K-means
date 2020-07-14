import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import binKMeans

#处理数据
df=pd.read_csv(file)
df.head()    #查看数据

data=df.loc[:,['商业得分','休闲娱乐得分','文化艺术得分','创业公司得分','科研机构得分']]
data.rename(columns={'商业得分':'Comm','休闲娱乐得分':'Ente','文化艺术得分':'Cult',
                     '创业公司得分':'Corp','科研机构得分':'Rese'},inplace=True)


#选择k值
K=[]
SSE=[]
for i in range(2,15):     #选择k值范围在2到15之间
    k=i
    centList,clusterAssment=binKMeans(data,k)
    sse=(array(clusterAssment.sum(axis=0)))[0][1]  #当k为i时所有样本数据的误差平方和加总
    K.append(i)
    SSE.append(sse)
print( K,SSE)

kVar=pd.DataFrame({'k':K,'SSE':SSE})

import matplotlib.pyplot as plt
plt.plot(K, SSE, 'ro-', color='BLUE', linewidth=1, )
plt.xlabel('k')
plt.ylabel('SSE')
plt.show()
