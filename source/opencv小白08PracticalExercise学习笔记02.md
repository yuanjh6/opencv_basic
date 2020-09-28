# opencv小白08PracticalExercise学习笔记02
## 10使用Hu矩进行形状匹配
Hu矩（或者更确切地说是Hu矩不变量）是使用对图像变换不变的中心矩计算的一组7个变量。事实证明，前6个矩不变量对于平移，缩放，旋转和映射都是不变的。而第7个矩会因为图像映射而改变。  
OpenCV中，我们HuMoments()用来计算输入图像中的Hu矩。  
```
_,im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
moment = cv2.moments(im)
huMoments = cv2.HuMoments(moment)
```
基于matchShapes函数计算两个图形之间的距离  
![](_v_images/20200712002309826_67565875.png =512x)  
```
m2 = cv2.matchShapes(im1,im2,cv2.CONTOURS_MATCH_I2,0)
```
您可以通过第三个参数（CONTOURS_MATCH_I1，CONTOURS_MATCH_I2或CONTOURS_MATCH_I3）使用三种b不同的距离。如果上述距离很小，则两个图像（im1和im2）相似。您可以使用任何距离测量。它们通常产生类似的结果。  
## 11基于OpenCV的二维码扫描器
核心代码  
```
qrDecoder = cv2.QRCodeDetector()
data,bbox,rectifiedImage = qrDecoder.detectAndDecode(inputImage)
```

## 12使用深度学习和OpenCV进行手部关键点检测
核心代码  
```
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
print("time taken by network : {:.3f}".format(time.time() - t))

points = []

for i in range(nPoints):
    probMap = output[0, i, :, :]
    probMap = cv2.resize(probMap, (frameWidth, frameHeight))

    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    if prob > threshold :
        cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        points.append((int(point[0]), int(point[1])))
    else :
        points.append(None)
```

## 13OpenCV中使用Mask R-CNN进行对象检测和实例分割
核心代码  
```
# For each frame, extract the bounding box and mask for each detected object
def postprocess(boxes, masks):
    numClasses = masks.shape[1]
    numDetections = boxes.shape[2]

    frameH = frame.shape[0]
    frameW = frame.shape[1]

    for i in range(numDetections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]
        if score > confThreshold:
            classId = int(box[1])

            # Extract the bounding box
            left top right bottom = xxx

            classMask = mask[classId]
            drawBox(frame, classId, score, left, top, right, bottom, classMask)

```
## 14使用OpenCV实现单目标跟踪
核心代码  
```
bbox = cv2.selectROI(frame, False)
ok = tracker.init(frame, bbox)

while True:
    ok, frame = video.read()
    if not ok:
        break

    ok, bbox = tracker.update(frame)

    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
```
## 15基于深度学习的目标跟踪算法GOTURN
核心代码:同上，只是模型变了，思路未变  
## 16使用OpenCV实现多目标跟踪Video
核心代码:和单目标类似，Tracker变成多个  
```
# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()
for bbox in bboxes:
multiTracker.add(createTrackerByName(trackerType), frame, bbox)

while cap.isOpened():
success, frame = cap.read()
if not success:
  break

# get updated location of objects in subsequent frames
success, boxes = multiTracker.update(frame)

# draw tracked objects
for i, newbox in enumerate(boxes):
  p1 = (int(newbox[0]), int(newbox[1]))
  p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
  cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
```
## 17基于卷积神经网络的OpenCV图像着色(略)

## 18Opencv中的单应性矩阵Homography(略)
Homography的应用-全景拼接  
```
# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
```
## 19使用OpenCV实现基于特征的图像对齐
OpenCV的图像对齐  
2.1 基于特征的图像对齐的步骤  
现在我们可以总结图像对齐所涉及的步骤。  
**Step1读图**  
我们首先在C ++中和Python中读取参考图像（或模板图像）和我们想要与此模板对齐的图像。  
**Step2寻找特征点**  
我们检测两个图像中的ORB特征。虽然我们只需要4个特征来计算单应性，但通常在两个图像中检测到数百个特征。我们使用Python和C ++代码中的参数MAX_FEATURES来控制功能的数量。  
**Step3 特征点匹配**  
我们在两个图像中找到匹配的特征，按匹配的评分对它们进行排序，并保留一小部分原始匹配。我们使用汉明距离（hamming distance）作为两个特征描述符之间相似性的度量。请注意，我们有许多不正确的匹配。  
**Step4 计算Homography**  
当我们在两个图像中有4个或更多对应点时，可以计算单应性。上一节中介绍的自动功能匹配并不总能产生100％准确的匹配。20-30％的比赛不正确并不罕见。幸运的是，findHomography方法利用称为随机抽样一致性算法（RANSAC）的强大估计技术，即使在存在大量不良匹配的情况下也能产生正确的结果。RANSAC具体介绍见：  
https://www.cnblogs.com/xingshansi/p/6763668.html  
https://blog.csdn.net/zinnc/article/details/52319716  
**Step5 图像映射**  
一旦计算出准确的单应性，我可以应用于一个图像中的所有像素，以将其映射到另一个图像。这是使用OpenCV中的warpPerspective函数完成的。  
