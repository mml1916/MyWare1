#coding:utf-8
from numpy import *
import operator 

#(1)计算已知类别数据集中的点与当前点之间的距离；
#(2)按照距离递增次序排序；
#(3)选取与当前点距离最小的走个点；
#(4)确定前灸个点所在类别的出现频率；
#(5)返回前女个点出现频率最高的类别作为当前点的预测分类。



#创建数据集和标签
def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels
#
#利用shape可输出矩阵的维度，即行数和列数。shape[0]和shape[1]分别代表行和列的长度
def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0] #4
	#拼贴函数tile ，从后往前拼贴为新数组, 之后与dataset数组做减法运算
	# [ [0,0],
	#	[0,0],
	#	[0,0],
	#	[0,0]]
	diffMat = tile (inX, (dataSetSize, 1)) - dataSet 
	#由各数组元素的平方构成新的数组
	sqDiffMat = diffMat **2
	#沿着横轴方向求和
	sqDistances = sqDiffMat.sum(axis = 1)
	#对各元素开根号
	distances = sqDistances**0.5
	#对数组或元组a进行升序排序， 返回的是升序之后的各个元素在原来a未升序之前的下标，即返回升序之后对应的下标数组
	sortedDistIndicies = distances.argsort()
	#创建字典
	classCount= {}
	for i in range(k):
	#选取与[0,0]距离最小的k个点，返回每个点的标签
		voteIlabel = labels[sortedDistIndicies[i]]
	#dict.get(key, default=None) key -- 字典中要查找的键。default -- 如果指定键的值不存在时，返回该默认值。本例中，即返回0
	#该句代码，实际上是在统计前K个点所在类别出现的频数
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	#sorted  为python内置排序函数，该语句为使用字典classCount的迭代器进行排列，按排列对象的第1个域（也即每个类型的频数）大小进行比较，降序排列
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]
	
def file2matrix(filename):
	fr = open(filename)
	#readlines()每次按行读取整个文件内容，将读取到的内容放到一个列表中，返回list类型，该列表可以由for... in ... 结构进行处理
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	#创建全0的多维数组
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
	#s为字符串，rm为要删除的字符序列
	#s.strip(rm)        删除s字符串中开头、结尾处，位于 rm删除序列的字符
	#当rm为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')
		line = line.strip()
	#Python split()通过指定分隔符对字符串进行切片，如果参数num 有指定值，则仅分隔 num 个子字符串
		listFromLine = line.split('\t')
	#多维数组赋值
		returnMat[index,:] = listFromLine[0:3]
	#将标签放入列表，int（lisFromLine[-1]）将字符串转换为int
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector

	
#准备数据：归一化数值（处理不同取值范围的特征值），方法使用当前值减去最小值，然后除以取值范围（83和85行）
def autoNorm(dataSet):
	#在数组第一维中比较，在列中选取最小值，而不是选取当前行的最小值，如a=np.array([1,5,3],[2,4,6]), 则a.min(0) = [1,2,3]
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals-minVals
	#创建一个和dataSet维度相等的全0数组
	normDataSet = zeros(shape(dataSet))
	#求dataSet数组第一维的维数
	m = dataSet.shape[0]
	#利用拼贴tile将minVals重复m次，详细用法参见有道云笔记numpy，之后和dataSet相减
	normDataSet = dataSet-tile(minVals,(m,1))
	#同上，相除
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet, ranges, minVals
	