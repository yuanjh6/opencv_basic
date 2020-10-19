# opencv小白01学习笔记01
## 学习目标
1,opencv能做什么，不能做什么

2,阅读代码，知道某种func后图片怎么样了

3,在图片相关机器学习算法预处理阶段，希望通过对图片的简单处理，达到提升训练效果的目的。

## OpenCV-Python教程:2.Images
https://www.jianshu.com/p/35712839830

·打开图片，显示，保存图片

·这些函数：cv2.imread(), cv2.imshow(), cv2.imwrite()

cv2.waitKey()是一个键盘绑定函数。它的参数是毫秒数，这个函数会等待任意键盘事件指定的毫秒时间。如果你点了任意键，这个程序继续。如果传入0，它会一直等待按键。它也可以设置成检测指定键，比如如果a被按了



cv2.destroyAllWindows()销毁所有的我们创建的窗口，如果你想销毁指定的窗口，使用函数cv2.destroyWindow()你可以传指定窗口的名字作为参数。

如果你使用64位的机器，你需要把k = cv2.waitKey(0) 这行换成：

```
k = cv2.waitKey(0) & 0xFF  
# 图集  
imgs = np.hstack([img,img2])  
# 展示多个  
cv.imshow("mutil_pic", imgs)  
```


## OpenCV-Python教程:3.视频
https://www.jianshu.com/p/562a936512ae

cap = cv2.VideoCapture(0)

cap.isOpened()

可以使用cap.get(propId)来访问这个视频的一些属性，propId是从0到18 的一个数。每个数字代表了视频的一个属性。

比如，我可以通过cap.get(3)和cap.get(4)来获得这一帧的宽度和高度。它默认会返回640x480，但是我想把值修改成320x240.只需要用ret = cap.set(3, 320)和ret = cap.set(4, 240)

```
fourcc = cv2.VideoWriter_fourcc(*'XVID')  
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  
out.write(frame)  
out.release()  
```
## OpenCV-Python教程:4.在OpenCV里的绘制函数
https://www.jianshu.com/p/b0adf093e15f

cv2.line(), cv2.circle(), cv2.rectangle(), cv2.ellipse(), cv2.putText()

这些函数里，你会发现一些通用的参数：

·img：你要画形状的图片

·color：形状的颜色。对于BGR，传一个元组进去，比如（255,0,0）是蓝色。 对于灰度图，传一个灰度值。

·thickness：线或者圆的粗细。如果传了-1给一个封闭图形比如圆，它会充满图形。默认的thickness = 1

·lineType: 线的类型，比如8-connected，反锯齿等。默认情况下是8-connected。cv2.LINE_AA是反锯齿，在曲线时很好看。

```
img = cv2.line(img,(0,0),(511,511),(255,0,0),5)  
img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)  
img = cv2.circle(img,(447,63),63,(0,0,255),-1)  
img = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)  
  
pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)  
pts = pts.reshape((-1,1,2))  
img = cv2.polylines(img,[pts],True,(0,255,255))  
```
注意：

如果第三个参数是False，你会得到一个连接所有点的图形，而不是一个封闭图形。

cv2.polylines()可以被用来画多条线，值需要建一个包含所有线的列表，然后把它传给函数就行了。所有线都会被独立绘制。这比每条线都调用一次cv2.line()更快更好的方法。



font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(img, 'OpenCV', (10,500), font, 4,(255,255,255), 2, cv2.CV_AA)



## OpenCV-Python教程:5.鼠标作为画笔
https://www.jianshu.com/p/e261346db440

```
cv2.setMouseCallback('image',draw_circle)  
def draw_circle(event, x, y, flags, param):  
    global ix, iy, drawing, mode  
  
    if event == cv2.EVENT_LBUTTONDOWN:  
        drawing = True  
        ix,iy = x,y  
    elif event == cv2.EVENT_MOUSEMOVE:  
        if drawing == True:  
            if mode == True:  
                cv2.rectangle(img, (ix, iy), (x, y),(0,255,0),-1)  
            else:  
                cv2.circle(img,(x,y),5,(0,0,255),-1)  
    elif event == cv2.EVENT_LBUTTONUP:  
        drawing = False  
        if mode == True:  
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)  
        else:  
            cv2.circle(img,(x,y),5,(0,0,255),-1)  
```
## OpenCV-Python教程:7.图片上的基本操作
https://www.jianshu.com/p/80efbe3880dc

```
>>>b,g,r = cv2.split(img)  
>>>img=cv2.merge((b,g,r))  
>>>b = img[:,:,0]  
```
假设你想把所有的红色像素变成0，你不用这么分割，你可以简单的使用Numpy索引，这样更快

```
>>>img[:,:,2]=0  
```
cv2.split()是一个成本很高的操作（执行时间），所以只在必要的时候使用。Numpy索引要更有效率，能用就用。



## OpenCV-Python教程:8.图片的算术运算
https://www.jianshu.com/p/4c4b4e651989

OpenCV和Numpy相加是不同的。OpenCV相加是一个渗透运算，而Numpy的相加是模运算。

```
>>>x = np.uint8([250])  
>>>y = np.uint8([10])  
>>>printcv2.add(x,y) ? ? ? # 250+10 = 260 => 255  
[[255]]  
>>>print x+y ? ? ? ?# 250+10 = 260 % 256 = 4  
[4]  
  
cv2.addWeighted()  
```

【Python——opencv篇】 bitwise_and、bitwise_not等图像基本运算及掩膜：https://blog.csdn.net/Lily_9/article/details/83143120

```
# set blue thresh  
lower_yellow=np.array([11,43,46])  
hsv = cv.cvtColor(cp, cv.COLOR_BGR2HSV)  
mask = cv.inRange(hsv, lower_yellow, upper_yellow)  
cv.imshow('Mask', mask)  
res = cv.bitwise_and(cp, cp, mask=mask)  
```

## OpenCV-Python教程:9.性能测量和改进技术
https://www.jianshu.com/p/205a7514641a

cv2.getTickCount 函数返回从一个参考时间（比如机器开机的时间）开始到这个函数被调用的时间之间的时钟循环数量。所以如果你在函数执行前调用一次，函数执行完调用一次，你就能得到函数执行用掉的时钟循环。

cv2.getTickFrequency函数返回时钟频率或者每秒钟的时钟循环数。所以要得到函数执行了多少秒



OpenCV的默认优化

很多OpenCV函数对SSE2, AVX等做了优化。当然也有未优化的代码。所以如果我们的系统支持这些特性，我们应该利用他们（基本上现在的主流处理器都支持）。在编译的时候是自动启用的。所以如果启用的话OpenCV执行的是优化的代码，你可以用cv2.useOptimized()来检查是否启用了，用cv2.setUseOptimized()来启用/禁用

Python标量运算时比Numpy标量运算要快的。所以对于包含1到两个元素的运算，Python标量要比Numpy数组要快。Numpy在数组尺寸有点大的时候占优势。

注意：

一般来说，OpenCV函数比Numpy函数要快，所以对于相同的运算，推荐优先使用OpenCV函数。但是，也有例外，特别是当Numpy操作views而不是复制的时候。



## OpenCV-Python教程:10.更改颜色空间
https://www.jianshu.com/p/65c1fbd8ae2a、

了解下列函数：cv2.cvtColor(), cv2.inRange()



变更色彩空间

```
# define range of blue color in HSV  
lower_blue = np.array([110,50,50])  
upper_blue = np.array([130,255,255])  
  
# Threshold the HSV image to get only blue colors  
mask = cv2.inRange(hsv,lower_blue,upper_blue)  
  
# Bitwise-AND mask and original image  
res = cv2.bitwise_and(frame,frame,mask=mask)  
  
>>>green = np.uint8([[[0,255,0]]])  
>>>hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)  
>>>print hsv_green  
[[[ 60 255 255]]]
```
