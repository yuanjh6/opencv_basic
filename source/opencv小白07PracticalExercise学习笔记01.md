# opencv小白07PracticalExercise学习笔记01
资料:OpenCV-Practical-Exercise:https://github.com/luohenyueji/OpenCV-Practical-Exercise

## 学习目的
1,OpenCV用法

2,各种机器学习场景涉及的opencv方法，问题解决思路等

3,各机器学习算法使用场景和特点


## 1基于深度学习识别人脸性别和年龄
核心代码

```
faceNet = cv.dnn.readNet(faceModel, faceProto)
frameFace, bboxes = getFaceBox(faceNet, frame)
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    
    gender = genderList[genderPreds[0].argmax()]
```

## 2人脸识别算法对比
核心代码

```
faceCascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
outOpencvHaar, bboxes = detectFaceOpenCVHaar(faceCascade, frame)
    faces = faceCascade.detectMultiScale(frameGray)
```

## 3透明斗篷
核心代码

```
img = np.flip(img,axis=1)#翻转

mask2 = cv2.inRange(hsv,lower_red,upper_red)# 0,1点阵

mask1 = mask1+mask2

# Refining the mask corresponding to the detected red color
mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
mask1 = cv2.dilate(mask1,np.ones((3,3),np.uint8),iterations = 1)#去除噪音
mask2 = cv2.bitwise_not(mask1)#取反

# Generating the final output
res1 = cv2.bitwise_and(background,background,mask=mask1)#从背景中获取布的部分
res2 = cv2.bitwise_and(img,img,mask=mask2)#从前景中获取布之外部分
final_output = cv2.addWeighted(res1,1,res2,1,0)#合并，布以外=前景，布内部＝背景，效果就是布变成透视的(隐身效果的布)
```
## 4OpenCV中的颜色空间

## 5基于深度学习的文本检测(略)
## 6基于特征点匹配的视频稳像(略)
## 7使用YOLOv3和OpenCV进行基于深度学习的目标检测Model
YOLO目标检测器首先，它将图像划分为13×13的单元格。这169个单元的大小取决于输入的大小。对于我们在实验中使用的416×416输入尺寸，单元尺寸为32×32。然后每个单元格作为一个边界框进行一次检测。对于每个边界框，网络还预测边界框实际包围对象的置信度，以分类的概率。大多数这些边界框都被消除了，因为它们的置信度很低，或者因为它们与另一个具有非常高置信度得分的边界框包围相同的对象。该技术称为非极大值抑制。

YOLOv3作者使YOLOv3比以前的作品YOLOv2更快，更准确。YOLOv3可以更好地进行多个尺度检测。他们还通过增加网络来改进网络。

```
modelConfiguration = "yolov3.cfg";
modelWeights = "yolov3.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
```
第一个设置，假如设置DEFAULT，默认设置的话，必须设置一个环境变量，并且变量的路径要是磁盘上一个文件夹，文件夹要存在，否则会警告或者报错。假如设置成OPENCV，会在用户名一个临时文件夹生成一些OPENCL的文件。建议设置为OPENCV，不用去配置环境变量

第二个设置，假如设置为CPU的话，速度较慢，通用性较好。设置为OPENCL的话，只能运行在inter的GPU上。假如电脑上有NVIDIA的话，会一直卡住，目前还没找到设置OPENCV运行哪块GPU的方法，没有在NVIDIA上的电脑上运行过。所以，为了确保GPU加速，不要在有NVIDIA电脑上运行


```
# Create a 4D blob from a frame.
blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

# Sets the input to the network
net.setInput(blob)

# Runs the forward pass to get output of the output layers
outs = net.forward(getOutputsNames(net))
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Remove the bounding boxes with low confidence
postprocess(frame, outs)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x ,center_y=xx
                width ,height , left,top = xx
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

```

## 8深度学习目标检测网络YOLOv3的训练(略)
## 9使用OpenCV寻找平面图形的质心	
核心代码

```
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray_image,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
	# calculate moments for each contour
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	
	cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
```



