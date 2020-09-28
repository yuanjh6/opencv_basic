# opencv小白03学习笔记03
## OpenCV-Python教程:21.轮廓：更多函数
https://www.jianshu.com/p/07312149e60a  
1.凸面缺陷   
OpenCV提供了现成的函数来做这个，cv2.convexityDefects().  
2.Point Polygon Test   
这个函数找到图像里的点和轮廓之间的最短距离。它返回的距离当点在轮廓外的时候是负值，当点在轮廓内是正值，如果在轮廓上是0  
3.匹配形状   
OpenCV提供一个函数cv2.matchShapes()来让我们可以比较两个形状，或者两个轮廓来返回一个量表示相似度。结果越低，越相似，它是根据hu矩来计算的。不同的计算方法在文档里有介绍。   

## OpenCV-Python教程:22.轮廓层级
https://www.jianshu.com/p/4daf1f7e69e0  
什么是层级?   
一般来说我们用cv2.findContours()函数来检测图像里的目标，有时候目标在不同的地方，但是在有些情况下，有些图形在别的图形里面，就像图形嵌套，在这种情况下，我们把外面那层图形叫做parent，里面的叫child。这样图形里的轮廓之间就有了关系。我们可以指定一个轮廓和其他之间的是如何连接的，这种关系就是层级。   
OpenCV里的层级表示   
每个轮廓有他自己的关于层级的信息，谁是他的孩子，谁是他的父亲等。OpenCV用一个包含四个值得数组来表示：[Next, Previous, First_Child, Parent]   
我们知道了层级，现在来看OpenCV里的轮廓获取模式，四个标志cv2.RETR_LIST, cv2.RETR_TREE, cv2.RETR_CCOMP, cv2.RETR_EXTERNAL表示啥？   
轮廓获取模式   
1.RETR_LIST   
这是最简单的一个，它获取所有轮廓，但是不建立父子关系，他们都是一个层级。所以，层级属性第三个和第四个字段（父子）都是-1，但是Next和Previous还是有对应值。  
2.RETR_EXTERNAL   
如果用这个模式，它返回最外层的。所有孩子轮廓都不要，我们可以说在这种情况下，只有家族里最老的会被照顾，其他都不管。所以在我们的图像里，有多少最外层的轮廓呢，有3个，contours 0,1,2   
3.RETR_CCOMP   
这个模式获取所有轮廓并且把他们组织到一个2层结构里，对象的轮廓外边界在等级1里，轮廓内沿（如果有的话）放在层级2里。如果别的对象在它里面，里面的对象轮廓还是放在层级1里，它的内沿在层级2.   
4.RETR_TREE   
最后，Mr.Perfect。它取回所有的轮廓并且创建完整的家族层级列表，它甚至能告诉你谁是祖父，父亲，儿子，孙子。。   
## OpenCV-Python教程:23.histogram
https://www.jianshu.com/p/c3f414646a50  
什么是histogram？它可以给出图像的密度分布的总体概念，它的x轴是像素值（0到255）y轴是对应的像素在图像里的数量。   
看histogram你可以得到对比度，亮度，密度分布等直观信息。  
1.OpenCV Histogram 计算  
2.Numpy里的Histogram计算  
OpenCV函数要比np.histogram()要快很多（40x）。所以还是用OpenCV函数。  
使用Mask  
我们使用cv2.calcHist()来找整个图的histogram。如果你想找某个区域的histogram，就创建一个你想要的区域是白色而其他地方是黑色的mask图像。  


## OpenCV-Python教程:24.histogram-2:histogram均衡
https://www.jianshu.com/p/7494f40e722b   
一张像素值被限制在一个特定值范围内的图像，比如，亮图被限制所有像素都是亮值。但是一个好的图片应该是有所有范围的像素。所以我们需要把histogram拉伸到两端，这就是histogram均衡。这个一般是用来提升图片的对比度。    
在面部识别里，在训练面部数据之前，面部的图片会做histogram均衡以让他们所有都是同样的亮度条件。   
OpenCV里的Histograms均衡    
OpenCV有一个函数可以做这个，cv2.equalizeHist()。它的输入是灰度图像，输出时我们的histogram均衡图像。   
```
img = cv2.imread('wiki.jpg',0)
equ = cv2.equalizeHist(img)
```
histogram均衡在图片的histogram被限制在一个范围内时很有用，它在histogram覆盖大范围时并不好用。  
对比度受限自适应直方图均衡    
我们第一个histogram均衡考虑的是图片全部对比度，在很多情况下，这不是好主意，   
背景对比度在均衡后确实提高了。但是对比两张图里的雕像，我们由于过度亮而导致失去了大部分面部信息。这是由于图像的histogram并不是像之前的图片那样限制在特定范围内。    
要解决这个问题，要用适应性histogram均衡。在这里，图像被分成小块，这些小块叫做瓷砖（瓷砖的大小在OpenCV里默认是8x8）。然后这些小块还和平常一样做均衡，所以在小块里，histogram是在小范围内的，如果有噪点，会被放大。要避免这个，要应用对比度限制。如果任何histogram 高于特定的对比度限制（OpenCV里默认是40），那些像素会被修剪掉并被无变化的放到其他然后再做histogram均衡。均衡后，要移除瓷砖边界的人工因素，要应用双线性插值。    
下面的代码显示了OpenCV里的CLAHE：  
```
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv2.imwrite('clahe_2.jpg',cl1)
```
## OpenCV-Python教程:25.Histograms-3:2d Histograms
https://www.jianshu.com/p/2f3fcabad43a   
在第一节里，我们计算和绘制了一维的histogram，它被叫做一维histogram是因为我们只拿了一个属性出来，像素的灰度强度值。而如果是二维histogram，你就要考虑两个属性了。一般来说对于彩色histogram两个属性是色调和饱和度的值。    
OpenCV里的2D histogram   
用cv2.calcHist()很简单，对于彩色histogram我们需要把图像从BGR转换成HSV。（记住，对于1维histogram，我们从BGR转成灰度），   
Numpy里的2D histogram   
Numpy也提供了函数np.histogram2d()。（记住，对于1维的是np.histogram()）    
## OpenCV-Python教程:26.Histogram 4 Histogram 向后投影
https://www.jianshu.com/p/31cd06b0bd6f   
用来在图片分割或者在图像里找感兴趣的目标，简单说来，它创建了一个和输入图像一样大小的图像，但是是单通道的。输出图像会有我们感兴趣的目标，但是它比其他部分更白。    
## OpenCV-Python教程:27.图像转换
https://www.jianshu.com/p/58c39dce2a7a   
傅里叶变换   
图像处理（5）--图像的傅里叶变换：https://blog.csdn.net/qq_33208851/article/details/94834614   
信号在频率域的表现   
在频域中，频率越大说明原始信号 变化速度越快；频率越小说明原始信号越平缓。当频率为0时，表示直流信号，没有变化。因此，频率的 大小反应了信号的变化快慢。高频分量解释信号的突变部分，而低频分量决定信号的整体形象。   
在 图像处理中，频域反应了图像在空域灰度变化剧烈程度，也就是图像灰度的变化速度，也就是图像的梯度大小。对图像而言，图像的边缘部分是突变部分，变化较 快，因此反应在频域上是高频分量；图像的噪声大部分情况下是高频部分；图像平缓变化部分则为低频分量。也就是说，傅立叶变换提供另外一个角度来观察图像， 可以将图像从灰度分布转化到频率分布上来观察图像的特征。书面一点说就是，傅里叶变换提供了一条从空域到频率自由转换的途径。对图像处理而言，以下概念非 常的重要：    
图像高频分量：图像突变部分；在某些情况下指图像边缘信息，某些情况 下指噪声，更多是两者的混合；   
低频分量：图像变化平缓的部分，也就是图像轮廓信息   
高通滤波器：让图像使低频分量抑制，高频分量通过   
低通滤波器：与高通相反，让图像使高频分量抑制，低频分量通过   
带通滤波器：使图像在某一部分 的频率信息通过，其他过低或过高都抑制   
还有个带阻滤波器，是带通的反。    
## OpenCV-Python教程:28.模板匹配
https://www.jianshu.com/p/53ef74b02f6a   
模板匹配是在一个大图里搜索和找模板图像位置的方法。OpenCV有个函数cv2.matchTemplate()来做这个。它吧模板图像在输入图像上滑动，对比模板和在模板图像下的输入图像块。它返回了一个灰度图像，每个像素表示那个像素周围和模板匹配的情况。    
如果输入图像大小是WxH而模板图像大小是wxh，输出图像的大小是（W-w+1， H-h+1）。当你得到了结果，你可以用cv2.minMaxLoc()函数来找最大最小值。把它作为矩形左上角，w，h作为矩形的宽和高。矩形是你的模板区域。    
注意：   
如果你用cv2.TM_SQDIFF作为比较方法，最小值是最匹配。    
     
尝试所有的比较方法   
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']    

模板匹配多个目标    
在前面我们搜索了messi的脸，目标只在图像里出现了一次，假设你要搜的东西在图像里出现多次，cv2.minMaxLoc()不会给你所有的位置。在这种情况下，我们会使用阈值，在这个例子里，我们使用超级玛丽的截图来找金币。    
```
img_rgb = cv2.imread('mario.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('mario_coin.png',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('res.png',img_rgb)
```
## OpenCV-Python教程:29.霍夫线变换
https://www.jianshu.com/p/722b5ac8773d   
霍夫变换（主要说明检测直线及圆的原理）:https://blog.csdn.net/weixin_40196271/article/details/83346442   
霍夫变换是图像处理中从图像中识别几何形状的基本方法之一，应用很广泛，也有很多改进算法。主要用来从图像中分离出具有某种相同特征的几何形状（如，直线，圆等）。最基本的霍夫变换是从黑白图像中检测直线(线段)。    
函数cv2.HoughLines()  

