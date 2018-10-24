# This code sample using onnxruntime packege by Microsoft (It's only provided for Linux)
#

import numpy as np
import onnxruntime as rt
from PIL import Image,ImageDraw
sess = rt.InferenceSession("tiny_yolov2/model.onnx")
input_name = sess.get_inputs()[0].name

img = Image.open('test.jpg')
img = img.resize((416, 416)) #for tiny_yolov2

X = np.asarray(img)
X = X.transpose(2,0,1)
X = X.reshape(1,3,416,416)
print(X.shape)

print(input_name)
out = sess.run(None, {input_name: X.astype(np.float32)})
print(out[0].shape)
out = out[0][0]

numClasses = 20
anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))

def softmax(x):
	scoreMatExp = np.exp(np.asarray(x))
	return scoreMatExp / scoreMatExp.sum(0)

clut = [(0,0,0),(255,0,0),(255,0,255),(0,0,255),(0,255,0),(0,255,128),
		(128,255,0),(128,128,0),(0,128,255),(128,0,128),
		(255,0,128),(128,0,255),(255,128,128),(128,255,128),(255,255,0),
		(255,0,128),(128,0,255),(255,128,128),(128,255,128),(255,255,0),
		]
print(len(clut))
label = ["aeroplane","bicycle","bird","boat","bottle",
         "bus","car","cat","chair","cow","diningtable","dog","horse",
         "motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]


draw = ImageDraw.Draw(img)
for cy in range(0,13):
	for cx in range(0,13):
		for b in range(0,5):
			channel = b*(numClasses+5)
			tx = out[channel  ][cy][cx]
			ty = out[channel+1][cy][cx]
			tw = out[channel+2][cy][cx]
			th = out[channel+3][cy][cx]
			tc = out[channel+4][cy][cx]

			x = (float(cx) + sigmoid(tx))*32
			y = (float(cy) + sigmoid(ty))*32
			
			w = np.exp(tw) * 32 * anchors[2*b  ]
			h = np.exp(th) * 32 * anchors[2*b+1]	
			
			confidence = sigmoid(tc)

			classes = np.zeros(numClasses)
			for c in range(0,numClasses):
				classes[c] = out[channel + 5 +c][cy][cx]
			classes = softmax(classes)
			detectedClass = classes.argmax()

			if 0.5< classes[detectedClass]*confidence:
				color =clut[detectedClass]
				print(detectedClass,label[detectedClass],classes[detectedClass]*confidence)
				x = x - w/2
				y = y - h/2
				draw.line((x  ,y  ,x+w,y ),fill=color)
				draw.line((x  ,y  ,x  ,y+h),fill=color)
				draw.line((x+w,y  ,x+w,y+h),fill=color)
				draw.line((x  ,y+h,x+w,y+h),fill=color)

img.save("result.png")
