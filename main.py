from ops import *

make_a_photo()
image = cv2.imread('images/sample.jpg')
orig = image.copy()

(origH, origW) = image.shape[:2]

rW = origW / float(newW)
rH = origH / float(newH)

image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

(rects, confidences) = decode_predictions(scores, geometry)
boxes = non_max_suppression(np.array(rects), probs=confidences)

results = []
index = 1

for (startX, startY, endX, endY) in boxes:
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)

	dX = int((endX - startX) * (padding - 0.03))
	dY = int((endY - startY) * (padding + 0.1))

	startX = max(0, startX - dX)
	startY = max(0, startY - dY)
	endX = min(origW, endX + (dX * 2))
	endY = min(origH, endY + (dY * 2))

	roi = orig[startY:endY, startX:endX]

	start = time.time()
	config = ("-l eng --oem 1 --psm 3")
	text = pytesseract.image_to_string(roi, config=config)
	end = time.time()
	print("time till recognition of word nr {} : {}".format(index, end-start))
	results.append(((startX, startY, endX, endY), text))
	index += 1

results = sorted(results, key=lambda r:r[0][1])

print('\n{} words were detected\n'.format(len(results)))

with open('output.txt','a') as f:
	f.write('///')

for ((startX, startY, endX, endY), text) in results:
	print("========")
	print("{}\n".format(text))

	text = "".join([c if ord(c) < 123 and ord(c) > 96 else '' for c in text]).strip()
	output = orig.copy()
	cv2.rectangle(output, (startX, startY), (endX, endY),
		(0, 0, 255), 2)
	cv2.putText(output, text, (startX, startY - 20),
		cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)
	with open('output.txt', 'a') as f:
		f.write(text)
		f.write('.')
	index += 1

	cv2.imshow("Text Detection", output)
	cv2.waitKey(0)

with open('output.txt','a') as f:
	f.write('\n')

