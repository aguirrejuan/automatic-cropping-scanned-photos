import numpy as np
import cv2, os, argparse, datetime, errno, math, multiprocessing
from concurrent.futures import ThreadPoolExecutor
import imutils

DEG_TO_RAD = math.pi / 180
EXTS = [".jpg", ".jpeg", ".png", ".bmp" ] # Image extensions to look for.
MAX = 255 # Thresholded max value (white).

parser = argparse.ArgumentParser(description="Scanned image cropper." +
			"\nProcess scanned images to find photos inside them." +
			"\nOrients and crops photos found in the image scan." +
			"\nProcesses all images found in the input directory, and" +
			"\nwrites all found and processed photos in the output directory." +
			"\nCan process multiple photos in a single scan.",
			formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir', '-d', type=str, default="./",
					help="Specify the location of the pictures to process.")
					
parser.add_argument('--odir', '-o', type=str, default="./output/",
					help="Specify where to save the processed scans.")
					
parser.add_argument('--num-threads', '-n', dest='threads', type=int, default=0,
					help="Number of threads to use." +
					"\n0 = system number of cores.")
					
parser.add_argument('--pic-size-diff', '-s', type=float, dest='scale', default=0.80,
					help="The approximate size difference between scanned images, as a percent." +
					"\nSet lower if images are of varying sizes." +
					"\nRange: [0.0,1.0]" )
					
parser.add_argument('--thresh', '-t', type=int, dest='thresh', default=230,
					help="Sets the threshold value when determining photo edges." +
					"\nUse higher values for brighter images. Lower for tighter cropping." +
					"\nRange [0,255]")
					
parser.add_argument('--photos-per-scan', '-i', type=int, dest='num_scans', default=1,
					help="Number of photos to look for per scanned image.")
					
parser.add_argument('--blur', '-b', type=int, dest='blur', default=9,
					help="How much blur to apply when processing." +
					"\nDifferent values may effect how well scans are found and cropped." +
					"\nMust be odd number greater than 1.")
parser.add_argument('--append-datetime', '-a', dest='useDatetime', action='store_true',
					help="Append the current date to the start of output image files.")
args = parser.parse_args()

THREADS = args.threads
THRESH = args.thresh
BLUR = args.blur
NUM_SCANS = args.num_scans
IM_SCALE = args.scale
IMDIR = args.dir
OUTDIR = args.odir
SHOULD_APPEND_DATETIME = args.useDatetime

if THREADS == 0 :
	THREADS = multiprocessing.cpu_count()

ERRORS = 0 # Total number of errors encountered.
IMAGES = 0 # Total number of image files processed.
SCANS  = 0 # Total number of images found in all scans.
	
# Try making the output directory.
try:
	os.makedirs(OUTDIR)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise

def getDatetime():
	return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def openImage(dir, file):
	path = os.path.join(dir, file)
	img = cv2.imread(path)
	# if img == None:
	# 	print("Error: Failed to open image at path: "+path)
	# 	global ERRORS
	# 	ERRORS += 1
	img = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, None, value =(255,255,255))
	return img

def writeImage(dir, fileName, img):
	path = os.path.join(dir, fileName)
	print(img.shape, path)
	success = cv2.imwrite(path, img)
	if not success:
		print("Error: Failed to write image "+fileName+" to file.")
		global ERRORS
		ERRORS += 1
		return False
	print("Wrote image to: " + path)
	return True

def writeScans(dir, fileName, scans):
	if len(scans) == 0:
		print("Warning: No scans were found in this image: "+fileName)
		global ERRORS
		ERRORS += 1
		return
	name, ext = os.path.splitext(fileName)
	if SHOULD_APPEND_DATETIME:
		name = "{}_{}".format(getDatetime(), name)
	num = 0
	for scan in scans:
		f = "{}_{}{}".format(name, num, ext)
		writeImage(dir, f, scan)
		num += 1

def getAveROISize(candidates):
	if len(candidates) == 0:
		return 0
	av = 0
	for roi in candidates:
		av += cv2.contourArea(roi[0])
	return av / len(candidates)
	
# Find regions of interest in the form [rect, box-contour].
# Attempts to find however many scans we're looking for in the image.
def getROI(contours):
	roi = []
	for contour in contours:
		rect = cv2.minAreaRect(contour)
		box = cv2.boxPoints(rect)
		roi.append([box, rect])
	
	roi = sorted(roi, key=lambda b: cv2.contourArea(b[0]), reverse=True)
	candidates = []
	for b in roi:
		if len(candidates) >= NUM_SCANS:
			break
		#if cv2.contourArea(b[0]) >= getAveROISize(candidates)*IM_SCALE:
		candidates.append(b)
	return candidates

def rotateImage(img, angle, center):
	(h, w) = img.shape[:2]
	mat = cv2.getRotationMatrix2D((int(center[0]),int(center[1])), angle, 1.0)
	return cv2.warpAffine(img, mat, (w,h), flags=cv2.INTER_LINEAR)

def rotateBox(box, angle, center):
	rad = -angle * DEG_TO_RAD
	sine = math.sin(rad)
	cosine = math.cos(rad)
	rotBox = []
	for p in box:
		p[0] -= center[0]
		p[1] -= center[1]
		rot_x = p[0] * cosine - p[1] * sine
		rot_y = p[0] * sine   + p[1] * cosine
		p[0] = rot_x + center[0]
		p[1] = rot_y + center[1]
		rotBox.append(p)
	return np.array(rotBox)

def getCenter(box):
	x_vals = [i[0] for i in box]; y_vals = [i[1] for i in box]
	cen_x = (max(x_vals) - min(x_vals))//2  + min(x_vals)
	cen_y = (max(y_vals) - min(y_vals))//2 + min(y_vals)
	return (cen_x, cen_y)



def rotate_image(img, angle, center):
	"""
	Rotates an image (angle in degrees) and expands image to avoid cropping
	"""
	
	mat = img 
	height, width = mat.shape[:2] # image shape has 3 dimensions
	image_center = center
	rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
	# rotation calculates the cos and sin, taking absolutes of those.
	abs_cos = abs(rotation_mat[0,0]) 
	abs_sin = abs(rotation_mat[0,1])
	# find the new width and height bounds
	bound_w = int(height * abs_sin + width * abs_cos)
	bound_h = int(height * abs_cos + width * abs_sin)
	# subtract old image center (bringing image back to origo) and adding the new image center coordinates
	rotation_mat[0, 2] += bound_w/2 - image_center[0]
	rotation_mat[1, 2] += bound_h/2 - image_center[1]
	# rotate image with the new bounds and translated rotation matrix
	rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
	return rotated_mat, rotation_mat

def rotate_box(box, angle, center):
	"""
	Rotates an image (angle in degrees) and expands image to avoid cropping
	"""
	width =int(np.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2))
	height =int(np.sqrt((box[1][0]-box[2][0])**2 + (box[1][1]-box[2][1])**2))
	image_center = center
	rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
	# rotation calculates the cos and sin, taking absolutes of those.
	abs_cos = abs(rotation_mat[0,0]) 
	abs_sin = abs(rotation_mat[0,1])
	# find the new width and height bounds
	bound_w = int(height * abs_sin + width * abs_cos)
	bound_h = int(height * abs_cos + width * abs_sin)
	# subtract old image center (bringing image back to origo) and adding the new image center coordinates
	rotation_mat[0, 2] += bound_w/2 - image_center[0]
	rotation_mat[1, 2] += bound_h/2 - image_center[1]
	# rotate image with the new bounds and translated rotation matrix

	rotBox = cv2.transform(np.array([box]), rotation_mat)[0]
	return rotBox



# Rotate and crop the candidates.
def clipScans(img, candidates):
	scans = []
	for roi in candidates:
		rect = roi[1]
		box = np.intp(roi[0])
		angle = rect[2]
		if angle < -45:
			angle += 90
		center = getCenter(box)
		center = int(center[0]), int(center[1])

		
		#rotBox = rotate_box(box,angle,center)
		rotIm, M = rotate_image(img, angle, center)
		rotBox = cv2.transform(np.array([box]), M)[0]

		x_vals = [i[0] for i in rotBox]
		y_vals = [i[1] for i in rotBox]
		try:
			new_image = rotIm[min(y_vals):max(y_vals), min(x_vals):max(x_vals)]
			scans.append(new_image)
		except IndexError as e:
			print("Error: Rotated image is out of bounds!\n" +
				"Try straightening the picture, and moving it away from the scanner's edge.", e)
			global ERRORS
			ERRORS += 1
	return scans
	
def findScans(img):
	blur = cv2.medianBlur(img, BLUR)
	grey = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
	#_, thr = cv2.threshold(grey, THRESH, MAX, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	ret, thr = cv2.threshold(grey,
    						0,  # threshold value, ignored when using cv2.THRESH_OTSU
    						255,  # maximum value assigned to pixel values exceeding the threshold
    						cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # thresholding type
	
	thr = imagem = cv2.bitwise_not(thr)
	
	im_floodfill = thr.copy()
	h, w = thr.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)
 
	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255);
	
	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)
	
	# Combine the two images to get the foreground.
	thr = thr | im_floodfill_inv

	contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	contours = [contour for contour in contours if cv2.contourArea(contour) > 10000]
	# for i,contour in enumerate(contours):
	# 	rect = cv2.minAreaRect(contour)
	# 	box = cv2.boxPoints(rect)
	# 	box = np.intp(box)
	# 	cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
	# 	print(cv2.contourArea(contour), 'dasdasdas')
		


	#cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
	# cv2.imwrite('dasd.png', img)
	roi = getROI(contours)
	scans = clipScans(img, roi)

	return scans

def processFile(file):
	global IMDIR, OUTDIR, IMAGES, SCANS
	print(file)
	img = openImage(IMDIR, file)
	print(img.shape)
	scans = findScans(img)
	writeScans(OUTDIR, file, scans)
	IMAGES += 1
	SCANS += len(scans)
	
#--------------------------------------------------------------------

# with ThreadPoolExecutor(max_workers=THREADS) as executor:
# 	for file in [f for f in os.listdir(IMDIR) if f.endswith(tuple(EXTS))]:
# 		executor.submit(processFile, file)


for file in [f for f in os.listdir(IMDIR) if f.endswith(tuple(EXTS))]:
	processFile(file)

print("\n-----------------------------------------------------")
print("{} pictures found in {} scan files.".format(SCANS, IMAGES))
print("Program completed with {} errors and warnings.".format(ERRORS))
