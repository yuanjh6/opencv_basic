# opencv小白04学习笔记04
## OpenCV-Python教程:30.霍夫圆变换
https://www.jianshu.com/p/a795171f8092

函数是cv2.HoughCircles()


## OpenCV-Python教程:31.分水岭算法对图像进行分割
https://www.jianshu.com/p/de81d6029235



## OpenCV-Python教程:32.使用GrabCut算法分割前景
https://www.jianshu.com/p/117f66320589

开始用户画一个矩形方块把前景图圈起来，前景区域应该完全在矩形内，然后算法反复进行分割以达到最好效果。但是有些情况下，分割的不是很好，比如把前景给标称背景了等。在这种情况下用户需要再润色，就在图像上有缺陷的点给几笔。这几笔的意思是说“嘿，这个区域应该是前景，你把它标成背景了，下次迭代改过来”或者是反过来。那么在下次迭代，结果会更好。

## OpenCV-Python教程:33.特征检测和描述
https://www.jianshu.com/p/f222200a5769


##  OpenCV-Python教程:34.Harris角点检测
https://www.jianshu.com/p/9c7cf8ea183f

cv2.cornerHarris(gray,2,3,0.04)

有时候，你可能需要更准确的找到角，OpenCV用函数cv2.cornerSubPix() 通过亚像素精度精炼角点检测。下面是例子，我们需要先找到harris角，然后我们把这些角的质心传进去（可能有一堆点在角上，我们找他们的质心）来精炼他们。

```
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
```

## OpenCV-Python教程:35.Shi-Tomasi 角点检测和特征跟踪
https://www.jianshu.com/p/163ff90e35a9

OpenCV有个函数cv2.goodFeaturesToTrack()。它会用Shi-Tomasi方法（或者Harris角点检测，你可以指定）找到N个最强的角。输入图像仍然应该是灰度图。然后你指定你想找到的角的数量，接着指定质量级别，值介于0和1之间，指明了角的最小质量。之后我们提供角之间的最小欧几里得距离。

通过所有这些信息，函数可以在图像里找角。所有低于质量级别的角被拒绝。然后它会根据质量降序对剩下的角排序。然后函数取第一个最强的角，把周围的最小距离内的所有角都扔掉，然后返回N个最强的角。


## OpenCV-Python教程:36.SIFT（尺度不变特征变换）
https://www.jianshu.com/p/c0379c931e74

在SIFT算法里主要有四步：

```
1.尺度空间极值检测
2.关键点本地化
3.方向分配
4.关键点描述
5.关键点匹配
```

OpenCV里的SIFT

sift.detect()函数找到图像的关键点，你可以传一个掩图给它，如果你只想在图像的一个部分内搜索的话。每个关键点是一个特殊的结构，这些结构有很多属性，比如(x,y)坐标，有意义的邻居的大小，指定它们方向的角度，指定关键点力量的响应等。

OpenCV也提供了cv2.drawKeyPoints()函数来在关键点的位置画上小圆圈，如果你传入一个标志位，cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS，它会画一个和关键点大小一样的圆，还会显示出它的方向。

计算描述，OpenCV提供了两个方法：

1.由于你已经找到关键点了，你可以调用sift.computer()来计算我们找到的关键点的描述，比如： kp, des = sift.compute(gray, kp)

2.如果你没找关键点，直接用sift.detectAndCompute()一次直接找到关键点和描述。

```
sift=cv2.SIFT()
kp,des=sift.detectAndCompute(gray,None)
```
这里kp是关键点的列表，des是形状数组


## OpenCV-Python教程:37.SURF(加速稳健特征）
https://www.jianshu.com/p/6aee43c6f2ac

```
# Again compute keypoints and check its number.
>>> kp, des = surf.detectAndCompute(img,None)
>>> print len(kp)
47
```
这比50少了，我们画出它来

```
>>>img2=cv2.drawKeypoints(img,kp,None,(255,0,0),4)
>>>plt.imshow(img2),plt.show()
```

## OpenCV-Python教程:38.FAST角点检测算法
https://www.jianshu.com/p/d778e3be32ff

我们看到了一些特征检测算法，他们很多都不错，但是从实时应用的角度看，他们都不够快，一个最好的例子是SLAM（同步定位与地图创建）移动机器人没有足够的计算能力。

作为解决方案，**FAST(加速切片测试特征）算法**被提出。


## OpenCV-Python教程:39.BRIEF
https://www.jianshu.com/p/a22b82f1df5f

BRIEF在这个时候出现了，它提供了直接找到二进制字符串而不找描述子的简便办法。它取被平滑过的图像块，选择nd(x,y)集合位置对，然后在这些位置对上做像素强度对比，比如，设第一个位置对为p和q，如果I(p) < I(q)，那么它的结果是1,否则是0，这用在所有nd个位置对，得到nd维的位串。

这里nd可以是128,256或者512.OpenCV支持所有这些，但是默认是256（OpenCV用字节表示，所以就是16，32和64字节）。当你得到这个，你可以使用Hamming距离来匹配这些描述子

一个重要的点是**BRIEF是一个特征描述子，它不提供任何方法来找特征**，所以你还得使用别的特征描述子比如SIFT，SURF等，论文推荐使用CenSurE，是个快速检测器，BRIEF甚至和CenSurE比SURF还好点。

简单说，**BRIEF是一个快速的特征描述子计算和匹配方法。它提供了高识别率**，除非是大的平面旋转  