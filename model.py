import numpy as np
import pandas as pd

#用二分k均值计算每个地块的复合类型
#计算两个向量欧氏距离
def distElud(vecA,vecB):
    return sqrt(sum(power((vecA - vecB),2)))

#二分k均值方法
def binKMeans(dataSet, k, distMeas = distElud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    # 创建一个初始簇, 取每一维的平均值
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]  # 记录有几个簇
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2
    while len(centList) < k:
        lowestSSE = inf
    #     # 找到对所有簇中单个簇进行2-means可以是所有簇的sse最小的簇
        for i in range(len(centList)):
            # 属于第i簇的数据
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # print ptsInCurrCluster
            # 对第i簇进行2-means
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 第i簇2-means的sse值
            sseSplit = sum(splitClustAss[:, 1])
            # 不属于第i簇的sse值
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i), 1])
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseNotSplit + sseSplit
        # 更新簇的分配结果
        #新增的簇编号
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        #另一个编号改为被分割的簇的编号
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit  #
        # 更新被分割的的编号的簇的质心
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        # 添加新的簇质心
        centList.append(bestNewCents[1, :].tolist()[0])
        # 更新原来的cluster assment
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment
