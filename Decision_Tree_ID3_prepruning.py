# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 20:43:08 2019
本程序实现决策树的预剪枝
数据集为西瓜书表4.2，划分为训练集与验证集
参考资料：周志华《机器学习》
@author: yanji
"""
import Decision_Tree_ID3 as TreeID3
import Decision_Tree_Visual as TreeVisual
import pandas as pd

#读取训练集
watermelonTra=pd.read_csv('watermelon2Training.csv',encoding='gbk')
#读取验证集
watermelonVal=pd.read_csv('watermelon2Validation.csv',encoding='gbk')

#基于训练集创建未剪枝决策树
treeOriginal=TreeID3.creatDecisionTree(watermelonTra)
#可视化未剪枝决策树
TreeVisual.createTree(treeOriginal,'未剪枝决策树')

#判断进行属性划分时是否使得验证集精度进行增加,即验证集中判断正确的样例个数是否增加
def precisionRaiseJudge(chosenFeature,dfdata,valdata):
    #此时训练集与验证集已经是上层决策树分完类的子集
    valNum=valdata.shape[0]
    typeColval =valdata.shape[1]-1
    #不划分，计算验证集预测正确个数
    numUnclassify=0;
    typedfdata=TreeID3.typeMajority(dfdata)
    for rowNum in range(valNum):
        if valdata.iloc[rowNum,typeColval]==typedfdata:
            numUnclassify+=1
    #划分，计算验证集预测正确个数
    numclassify=0
    chosenFeatureValueCount=dict(dfdata.loc[:,chosenFeature].value_counts())
    typedfdataClassify={}
    for key in chosenFeatureValueCount.keys():
        #判定各个属性取值对应的类
        typedfdataClassify[key]=TreeID3.typeMajority(TreeID3.splitDataset(dfdata,chosenFeature,key))
    #print(typedfdataClassify)
    for rowNum in range(valNum):
        featureValue = valdata.iloc[rowNum][chosenFeature]
        typeValue = valdata.iloc[rowNum,typeColval]
        for key,value in typedfdataClassify.items():
            if (featureValue == key and typeValue == value):
                numclassify+=1
    #如果划分后的验证集预测个数没有大于划分前个数，则不划分
    if numclassify>numUnclassify:
        return True
    else:
        return False
    
#创建预剪枝决策树
def createPrepruningTree(dfdata,valdata):
    #首先判断样本集是否为同一类别以及是否还能进行属性划分
    if (dfdata.shape[1]==1 or len(dfdata.iloc[:,dfdata.shape[1]-1].unique())==1):
        return TreeID3.typeMajority(dfdata) 
    bestFeature = TreeID3.chooseBestFeatureToSplit(dfdata)   #选择最佳划分属性    
    bestFeatureValueCount=dict(dfdata.loc[:,bestFeature].value_counts()) #统计该属性下的所有属性值
    #判断是否应该对此节点进行划分
    if(precisionRaiseJudge(bestFeature,dfdata,valdata)==False):
        return TreeID3.typeMajority(dfdata) 
    decisionTree={bestFeature:{}}  #以字典形式创建决策树
    for key, value in bestFeatureValueCount.items():
        #以递归调用方式不断完善决策树
        decisionTree[bestFeature][key]=createPrepruningTree(TreeID3.splitDataset(dfdata,bestFeature,key),TreeID3.splitDataset(valdata,bestFeature,key))
    return decisionTree

#基于训练集与验证集创建预剪枝决策树 
treePrepruning=createPrepruningTree(watermelonTra,watermelonVal)
#预剪枝决策树可视化
TreeVisual.createTree(treePrepruning,'预剪枝决策树')    