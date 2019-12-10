import argparse
import pickle
from math import log

from openpyxl import load_workbook, Workbook

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--xlsx', help='need a xlsx file. \
                    e.g. --xlsx example.xlsx', dest='XLSX')
args = parser.parse_args()



def loadData(file):
    wb = load_workbook(file)
    ws = wb.active

    dataSet = []
    labels = None

    for i,row in enumerate(ws.iter_rows()):
        rowdata = [cell.value for cell in row]
        if i > 1:
            rowdata = rowdata[1:]
            c = rowdata[0]
            del rowdata[0]
            rowdata.append(c)
            dataSet.append(rowdata)
        elif i == 1:
            rowdata = rowdata[2:]
            for i,r in enumerate(rowdata):
                rowdata[i] = rowdata[i][1:-1]
            labels = rowdata
    
    return dataSet, labels

def cal_2class_impurity(dataSet, mode='gini'):
    totalNum = len(dataSet)
    labelNum = {}
    for data in dataSet:
        class_ = data[1]
        if class_ in labelNum:
            labelNum[class_] += 1
        else:
            labelNum[class_] = 1
    if mode == 'classCount':
        return labelNum
    if mode == 'gini':
        gini = 0
        for key in labelNum:
            p = labelNum[key] / totalNum
            gini += p * (1-p)
        return gini
    elif mode == 'entropy':
        ent = 0.0
        for key in labelNum:
            prob = float(labelNum[key]) /  totalNum
            ent -= prob * log(prob, 2)
        return ent

def continuousAttribute(featList):
    """
    input 7 values, output 8 values
     1 2 3 4 5 6 7
    ^ ^ ^ ^ ^ ^ ^ ^
    """
    featList.sort()

    s = []
    s.append(featList[0] - (featList[0]+featList[1])/2)
    for i in range(0, len(featList)-1):
        s.append((featList[i]+featList[i+1])/2)
    s.append(featList[-1] + (featList[0]+featList[1])/2)
    return s
#------------------

def splitDataSet(dataSet, index, value):     
    subL = []
    subR = []                                     #创建返回的数据集列表
    for featVec in dataSet:                             #遍历数据集
        reducedFeatVec = featVec[:index]             #去掉axis特征
        reducedFeatVec.extend(featVec[index+1:])
        if featVec[index] <= value:
            subL.append(reducedFeatVec)
        elif featVec[index] > value:
            subR.append(reducedFeatVec)
    return subL, subR

def Dataset_class(dataSet):
    totalNum = len(dataSet)
    labelNum = {}
    for data in dataSet:
        class_ = data[1]
        if class_ in labelNum:
            labelNum[class_] += 1
        else:
            labelNum[class_] = 1

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 2                  #特征数量
    parentImpurity = cal_2class_impurity(dataSet)
    print('feature number: ',numFeatures) 
    print('parentImpurtiy: ', parentImpurity)
    

    bestInfoGain = 0.0                                #信息增益
    bestFeature = -1
    bestSplitValue = 0.0                                    #最优特征的索引值
    for i in range(2, numFeatures):                        #遍历所有特征
        #获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        splitPositions = continuousAttribute(featList)                    
        newImpurity = 100000.0
        newSplitValue = 0.0                 
        for value in splitPositions:     
            subL, subR = splitDataSet(dataSet, i, value)      #subDataSet划分后的子集

            probL = len(subL) / float(len(dataSet))        #计算子集的概率
            probR = len(subR) / float(len(dataSet))
            giniSplit = probL * cal_2class_impurity(subL) \
                      + probR * cal_2class_impurity(subR)

            #print(value, ' : ', len(subL), ' ', len(subR), ', giniSplit: ', giniSplit)
            if giniSplit < newImpurity:
                newImpurity = giniSplit
                newSplitValue = value
        

        infoGain = parentImpurity - newImpurity                     #信息增益
        #print("第%d个特征的增益为%.3f, gini split值為%.3f, 最低gini的split position為%f" % (i-2, infoGain, newEntropy, newSplitValue))           #打印每个特征的信息增益
        if (infoGain > bestInfoGain):                           #计算信息增益
            bestInfoGain = infoGain                             #更新信息增益，找到最大的信息增益
            bestFeature = i-2
            bestSplitValue = newSplitValue


    #print("最終：第%d个特征的增益为%.3f, 最低gini的split position為%f" % (bestFeature, bestInfoGain, bestSplitValue))  
    return bestFeature, bestSplitValue





def majorityCnt(classList):
    classCount = {}
    for vote in classList:  #统计classList中每个元素出现的次数
        if vote not in classCount.keys():classCount[vote] = 0   
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)     #根据字典的值降序排序
    return sortedClassCount[0][0]   #返回classList中出现次数最多的元素

def createTree(dataSet, labels, featLabels):
    classList = None
    try:
        classList = [example[1] for example in dataSet]
        if classList.count(classList[0]) == len(classList):         #如果类别完全相同则停止继续划分
            return classList[0]
        if len(dataSet[0]) == 1:                                    #遍历完所有特征时返回出现次数最多的类标签
            return majorityCnt(classList)
    except:
        print('classfication done.')
        return
    
    bestFeat, bestSplitValue = chooseBestFeatureToSplit(dataSet)
     
    print('best:', bestFeat) 
    print(labels)                #选择最优特征
    bestFeatLabel = labels[bestFeat]
    
    #print(bestSplitValue)                           #最优特征的标签
    featLabels.append(bestFeatLabel)
    bestFeatLabel +=  ' ' + str(bestSplitValue)
    myTree = {bestFeatLabel:{}}                                 #根据最优特征的标签生成树
    del(labels[bestFeat])
    print(myTree)
                                          
    #featValues = [example[bestFeat] for example in dataSet]     #得到训练集中所有最优特征的属性值
    #uniqueVals = set(featValues)                                #去掉重复的属性值
    subL, subR = splitDataSet(dataSet, bestFeat+2, bestSplitValue)
    left = '<= ' + str(bestSplitValue)
    right = '> ' + str(bestSplitValue)
    #print(len(subL))
    labelNum = cal_2class_impurity(subL, 'classCount')
    print(left, 'left',labelNum)
    #print(len(subR))
    labelNum = cal_2class_impurity(subR, 'classCount')
    print(right, 'right',labelNum)
    
    #print(len(dataSet))
    #print(len(subL))
    #print(len(subR))
    print('\n')
    if len(labels) > 0:                   
        myTree[bestFeatLabel]['left'] = createTree(subL, labels, featLabels)
        
        
    if len(labels) > 0:
        myTree[bestFeatLabel]['right'] = createTree(subR, labels, featLabels)
        
    
    
    return myTree

#------------------



#dataSet, labels, F_Name = loadData2(args.XLSX)
dataSet, labels = loadData(args.XLSX)





#featLabels = []
#myTree = createTree(dataSet, labels, featLabels)

#print(myTree)








