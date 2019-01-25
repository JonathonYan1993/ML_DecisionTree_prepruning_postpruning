# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 16:47:44 2019
本程序实现简单的决策树分类
决策树基于ID3决策树，以信息增益为准则选择划分属性
程序输出为决策树示意图
数据集为西瓜数据集2.0
参考资料：周志华《机器学习》
@author: yanji
"""


import pandas as pd
import math
import Decision_Tree_Visual

#读取西瓜数据集2.0
watermelon=pd.read_csv('watermelon2.csv',encoding='gbk')



#计算信息熵information entropy
def entropyCal(dfdata):
    dataSize=dfdata.shape[0] #数据集样本数
    colSize=dfdata.shape[1] #数据集属性个数（包括最后一列分类）
    typeCount=dict(dfdata.iloc[:,colSize-1].value_counts()) #统计数据集样本各个类别及其数目
    entropy=0.0
    for key in typeCount:
        p=float(typeCount[key])/dataSize
        entropy=entropy-p*math.log(p,2) #以2为底求对数
    return entropy    

#以某个属性的值划分数据集
def splitDataset(dfdata,colName,colValue):
    dataSize=dfdata.shape[0] #划分前数据集个数
    restData=pd.DataFrame(columns=dfdata.columns) #建立新的数据集，列索引与原数据集一样
    for rowNumber in range(dataSize):
        if dfdata.iloc[rowNumber][colName] == colValue:
            restData=restData.append(dfdata.iloc[rowNumber,:],ignore_index=True) #将划分属性等于该属性值的样本划给新数据集
    restData.drop([colName],axis=1,inplace=True) #去掉该属性列
    return restData

#选择当前数据集最好的划分属性
#ID3算法：以信息增益为准则选择划分属性
def chooseBestFeatureToSplit(dfdata):
    dataSize = dfdata.shape[0] #数据集样本个数
    numFeature = dfdata.shape[1]-1 #数据集属性个数
    entropyBase = entropyCal(dfdata) #划分前样本集的信息熵
    infoGainMax=0.0 #初始化最大信息熵
    bestFeature='' #初始化最佳划分属性
    for col in range(numFeature):
        featureValueCount=dict(dfdata.iloc[:,col].value_counts()) #统计该属性下各个值及其数目
        entropyNew=0.0
        for key, value in featureValueCount.items():
            #计算该属性划分下各样本集的信息熵加权和
            entropyNew+=entropyCal(splitDataset(dfdata,dfdata.columns[col],key))*float(value/dataSize) 
        infoGain=entropyBase-entropyNew #计算该属性下的信息增益
        if infoGain> infoGainMax:
            infoGainMax=infoGain
            bestFeature=dfdata.columns[col] #寻找最佳划分属性
    return bestFeature

#当叶节点样本已经无属性可划分了或者样本集为同一类别，这时采用多数表决法返回数量最多的类别
def typeMajority(dfdata):
    typeCount=dict(dfdata.iloc[:,dfdata.shape[1]-1].value_counts())
    return list(typeCount.keys())[0]        
            
#创建决策树
def creatDecisionTree(dfdata):
    #首先判断样本集是否为同一类别以及是否还能进行属性划分
    if (dfdata.shape[1]==1 or len(dfdata.iloc[:,dfdata.shape[1]-1].unique())==1):
        return typeMajority(dfdata) 
    bestFeature = chooseBestFeatureToSplit(dfdata)   #选择最佳划分属性    
    decisionTree={bestFeature:{}}  #以字典形式创建决策树
    bestFeatureValueCount=dict(dfdata.loc[:,bestFeature].value_counts()) #统计该属性下的所有属性值
    for key, value in bestFeatureValueCount.items():
        #以递归调用方式不断完善决策树
        decisionTree[bestFeature][key]=creatDecisionTree(splitDataset(dfdata,bestFeature,key))
    return decisionTree

#对新的样例进行分类预测
def classify(inputTree,valSple):
    firstStr = list(inputTree.keys())[0] #决策树第一个值，即第一个划分属性
    secondDict = inputTree[firstStr]
    for key in secondDict.keys():
        if(valSple[firstStr]==key): #该样本在该划分属性的值与决策树的对应判断
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],valSple) # 递归调用分类函数
            else:
                classLabel = secondDict[key]
    return classLabel

#创建西瓜数据集2.0的决策树        
watermelonDecisionTree=creatDecisionTree(watermelon)
#决策树可视化
Decision_Tree_Visual.createTree(watermelonDecisionTree,"ID3决策树_西瓜数据集2.0")






