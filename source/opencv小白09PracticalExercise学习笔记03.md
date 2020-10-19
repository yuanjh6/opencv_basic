# opencv小白09PracticalExercise学习笔记03
## 20使用OpenCV实现基于增强相关系数最大化的图像对齐(略)
## 21使用OpenCV的Eigenface
如何计算如何计算EigenFaces

要计算EigenFaces，我们需要使用以下步骤：

1）获取面部图像数据集：我们需要一组包含不同类型面部的面部图像。在这篇文章中，我们使用了来自CelebA的约200张图片。CelebA数据集见：
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

2）对齐和调整图像大小：接下来我们需要对齐和调整图像大小，以便在所有图像中眼睛的中心都是对齐的。这可以通过首先找到面部特征点来完成。在这篇文章中，我们使用了CelebA中提供的对齐图像。此时，数据集中的所有图像应该具有相同的大小。

3）创建数据矩阵：创建一个包含所有图像作为行向量的数据矩阵。如果数据集中的所有图像大小为100 x 100且有1000个图像，我们将拥有大小为30k x 1000的数据矩阵。

4）计算平均向量[可选]：在对数据执行PCA之前，我们需要减去平均向量。在我们的例子中，平均向量将是通过平均数据矩阵的所有行计算的30k×1行向量。使用OpenCV的PCA类不需要计算这个平均向量的原因是因为如果没有提供向量，OpenCV可以方便地计算我们的平均值。在其他线性代数包中可能不是这种情况。

5）计算主成分：通过找到协方差矩阵的特征向量来计算该数据矩阵的主成分。幸运的是，OpenCV中的PCA类为我们处理了这个计算。我们只需要提供数据矩阵，然后输出一个包含Eigenvectors的矩阵。

6）重塑特征向量以获得EigenFaces：如果我们的数据集包含大小为100 x 100 x 3的图像，那么如此获得的特征向量将具有30k的长度。我们可以将这些特征向量重塑为100 x 100 x 3图像以获得EigenFaces。

核心代码

```
data = createDataMatrix(images)#每个图像一行
mean, eigenVectors = cv2.PCACompute(data, mean=None, maxComponents=NUM_EIGEN_FACES)
averageFace = mean.reshape(sz)
eigenFaces = []; 

for eigenVector in eigenVectors:
	eigenFace = eigenVector.reshape(sz)
	eigenFaces.append(eigenFace)

# Display result at 2x size
output = cv2.resize(averageFace, (0,0), fx=2, fy=2)
cv2.imshow("Result", output)
```
## 22使用EigenFaces进行人脸重建(略)
## 23使用OpenCV获取高动态范围成像HDR(略)
## 24使用OpenCV进行曝光融合(略)
## 25使用OpenCV进行泊松克隆(略)
## 26基于OpenCV实现选择性搜索算法
目标检测与目标识别

目标识别算法Target Recognition**识别图像中存在哪些对象**。它将整个图像作为输入，并输出该图像中存在的对象的类标签和类概率。例如，类标签可以是“狗”，相关的类概率可以是97％。另一方面，目标检测算法**Target Detection不仅告诉您图像中存在哪些对象，还输出边界框（x，y，width，height）以表示图像内对象的位置**。

**所有目标检测算法的核心是物体识别算法**。假设我们训练了一个目标识别模型，该模型识别图像中的狗。该模型将判断图像中是否有狗。它不会告诉对象的位置。

为了定位物体，我们必须要选择图片的次区域（子块）然后对这些图片子块应用目标识别算法。目标的位置由目标识别算法返回的类概率较高的图像块的位置给出。

最直接的生成较小的次区域的方法为滑动窗口方法。然而，滑动窗口方法有许多限制。这些限制叫做候选区域算法克服了。**选择性搜索就是候选区域算法中最流行的一种**。

OpenCV有自带的SelectiveSearchSegmentation类

## 27在OpenCV下使用forEach进行并行像素访问(略,仅C)
OpenCV中有隐藏的功能，有时候并不是很有名。其中一个隐藏的功能是Mat类的forEach方法，它利用机器上的所有核心在每个像素上处理任何功能。


## 28基于OpenCV的GUI库cvui(略,仅C)
它是一个基于OpenCV绘图基元构建的跨平台GUI库，仅需使用头文件就可以搭建。除了OpenCV本身（您可能已经在使用）之外，它没有依赖关系。Cvui在C++下通过.h文件实现全部功能，在Python下直接提供.py文件。本文仅仅讲述cvui在C++下的构建，python通常用的少。

cvui遵循一行代码就可以在屏幕上产生一个UI组件的规则。cvui具有友好的C类API，没有类/对象和多个组件，例如， 跟踪栏，按钮，文字等等。


## 29使用OpenCV实现红眼自动去除
