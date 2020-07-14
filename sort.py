import numpy as np
import pandas as pd
import process
from model import binKMeans

centList,clusterAssment=binKMeans(data,4)

#classList=pd.DataFrame(clusterAssment)
#classList['Id']=data=df['Id']
#classList=pd.DataFrame(columns=['Id'],data=df['Id'])
#print(classList)
#print(list4[list4[0]==3])
def typeSort(clusterAssment,i): #i=0,1,2,3
    clusterData=pd.DataFrame(clusterAssment)
    clusterData['Id']=df['Id']  #增加区域id列
    typeAssment=clusterData[clusterData[0]==i]
    typeAssment.sort_values(by=[1],axis=0,ascending=False) #按照每个区域距聚类中心的SSE降序排列
    typeIdSort=typeAssment['Id']
    return typeIdSort
