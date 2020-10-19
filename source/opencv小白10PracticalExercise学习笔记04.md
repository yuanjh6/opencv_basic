# opencv小白10PracticalExercise学习笔记04
##	30使用OpenCV实现图像孔洞填充
##	31使用OpenCV将一个三角形仿射变换到另一个三角形
```
# Given a pair of triangles, find the affine transform.
warpMat = cv2.getAffineTransform( np.float32(tri1Cropped), np.float32(tri2Cropped) )

# Apply the Affine Transform just found to the src image
img2Cropped = cv2.warpAffine( img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

# Get mask by filling triangle
mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0);

img2Cropped = img2Cropped * mask
```
##	32使用OpenCV进行非真实感渲染
核心代码

```
# Edge preserving filter with two different flags.
imout = cv2.edgePreservingFilter(im, flags=cv2.RECURS_FILTER);
cv2.imwrite("edge-preserving-recursive-filter.jpg", imout);

imout = cv2.edgePreservingFilter(im, flags=cv2.NORMCONV_FILTER);
cv2.imwrite("edge-preserving-normalized-convolution-filter.jpg", imout);

imout = cv2.detailEnhance(im);
cv2.imwrite("detail-enhance.jpg", imout);

imout_gray, imout = cv2.pencilSketch(im, sigma_s=60, sigma_r=0.07, shade_factor=0.05);
cv2.imwrite("pencil-sketch.jpg", imout_gray);
cv2.imwrite("pencil-sketch-color.jpg", imout);

cv2.stylization(im,imout);
cv2.imwrite("stylization.jpg", imout);
```
##	33使用OpenCV进行Hough变换(略)
##	34使用OpenCV进行图像修复(略)
##	35使用Tesseract和OpenCV实现文本识别
如果使用tesseract，在实际工程tesseract错误率很高，识别率极差。一般需要对图像进行各种图像处理后再用tesseract识别，最后根据错误类型进行二次识别。tesseract的错误还是具有一定规律的。另外tesseract识别中文效果并不好，你要制作专门的中文训练集通过jTessBoxEditor.jar去训练它，但是整个制作流程较为复杂。具体见：

https://github.com/tesseract-ocr/tesseract/wiki/TrainingTesseract-4.00

tesseract要想有好的识别效果，就必须有大量的训练样本。但是tesseract对英文支持还是不错的。

```
imPath = 'image/computer-vision.jpg'
config = ('-l eng --oem 1 --psm 3')
im = cv2.imread(imPath, cv2.IMREAD_COLOR)
text = pytesseract.image_to_string(im, config=config)
print(text)
```

##	36使用OpenCV在视频中实现简单背景估计
核心代码

```
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

while(ret):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dframe = cv2.absdiff(frame, grayMedianFrame)
    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
```

##	37图像质量评价BRISQUE
##	38基于OpenCV的相机标定
##	39在OpenCV中使用ArUco标记的增强现实
aruco标记是放置在被成像的对象或场景上的基准标记。它是一个具有黑色背景和边界的二元正方形，其内部生成的白色图案唯一地标识了它。黑边界有助于他们更容易被发现。它们可以产生多种大小。根据对象大小和场景选择大小，以便成功检测。如果很小的标记没有被检测到，仅仅增加它们的大小就可以使它们的检测更容易。

想法是您打印这些标记并将其放置在现实世界中。您可以拍摄现实世界并独特地检测这些标记。如果您是初学者，您可能会在想这有什么用？让我们看几个用例。

在我们在帖子中分享的示例中，我们将打印的内容和标记放在相框的角上。当我们唯一地标识标记时，我们可以用任意视频或图像替换相框。当我们移动相机时，新图片具有正确的透视失真。

在机器人应用程序中，您可以将这些标记沿着配备有摄像头的仓库机器人的路径放置。当安装在机器人上的摄像头检测到一个这些标记时，它可以知道它在仓库中的精确位置，因为每个标记都有一个唯一的ID，我们知道标记在仓库中的位置。

##	40计算机视觉工具对比
2 适用于计算机视觉的MATLAB

```
2.1 为什么要使用MATLAB进行计算机视觉：优点
2.2 为什么不应该将MATLAB用于计算机视觉：缺点
```
3 适用于计算机视觉的OpenCV（C++）

```
3.1 为什么要使用OpenCV（C++）进行计算机视觉：优点
3.2 为什么不应该将OpenCV（C++）用于计算机视觉：缺点
```
4 适用于计算机视觉的OpenCV（Python）

```
4.1 为什么要使用OpenCV（Python）进行计算机视觉：优点
4.2 为什么不应该将OpenCV（Python）用于计算机视觉：缺点
```

##	41嵌入式计算机视觉设备选择
总而言之，Raspberry Pi，Jetson TK1和Jetson TX1明显领先于当今，拥有庞大的社区和公司。ODROID-C2是一匹黑马，可以替代Raspberry Pi。尽管如此，这个市场还处于新生阶段，有太多的大公司仍在努力在这个市场上有所作为。

实际上就个人经历而言，以深度学习为代表的人工智能技术最近遇到大挫折，深度学习存在许多瓶颈问题，计算机视觉技术也没有大的进展。现在工业应用上以caffe框架居多，实际也是云端/PC端/android端较多，嵌入式开发看看华为海思，英伟达的设备。树莓派搞搞研究挺好的，工业应用成本过高，其他的设备不建议使用。

##	42数码单反相机的技术细节(略)
