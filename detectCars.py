import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import math
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
from scipy.ndimage.measurements import label
import detectionHelpers as dh
import glob



def convert_color(img, conv='RGB2HSV'):
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
            cell_per_block, spatial_size, hist_bins, draw_img):


    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = dh.get_hog_features(ch1, orient, pix_per_cell, cell_per_block)
    hog2 = dh.get_hog_features(ch2, orient, pix_per_cell, cell_per_block)
    hog3 = dh.get_hog_features(ch3, orient, pix_per_cell, cell_per_block)
    boxes = []
    bbList = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = dh.bin_spatial(subimg, size=spatial_size)
            hist_features = dh.color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            #test_features = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,255,0),3)
                box = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw, ytop_draw+win_draw+ystart))
                bb = [xbox_left,ytop_draw+ystart,xbox_left+win_draw, ytop_draw+win_draw+ystart ]
                boxes.append(box)
                bbList.append(bb)

    return draw_img, boxes, bbList

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")



def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        for bb in box:
            
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
    
            if len(bb):
                heatmap[bb[0][1]:bb[1][1], bb[0][0]:bb[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes



def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap



def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    bbox2List = []
    for car_number in range(1, labels[1]+1):
       
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bbox2 = [np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)]
        bbox2List.append(bbox2)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img,bbox2List

def draw_bboxes(img, bBox, color):
    # Iterate through all detected cars
    for idx in range(0, len(bBox)):

        # Draw the box on the image
        bbMin = (bBox[idx,0],bBox[idx,1])
        bbMax = (bBox[idx,2],bBox[idx,3])
        cv2.rectangle(img, bbMin, bbMax, color, 2)
    # Return the image
    return img






# Features Parameters
orient, pix_per_cell, cell_per_block= 9, 8, 2 # HoG
spatial_size, hist_bins = (64, 64), 128 # Histogram

# Predictor Parameters
svc = pickle.load( open("svcModel_7_YCrCb_5000IMG.pkl", "rb" ) )
X_scaler = pickle.load( open("X_scaler_7_YCrCb_5000IMG.pkl", "rb" ) )

clip = VideoFileClip('project_video.mp4').subclip(8,9)
#clip = VideoFileClip('test_video.mp4')


# Processing Video in Lopp

count =0
auxBox = []
releaseBox=[]
for frames in clip.iter_frames():
    print(count)
    boxes = auxBox
    
    
    C_frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)
    C_frames_B  = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)

    heat = np.zeros_like(C_frames[:,:,0]).astype(np.float)
    draw_img = np.copy(C_frames)

     
    parameters = [(400, 500, 1),
                  (400, 550, 1.5),
                  (400, 700, 2.2)]

    
    for ystart, ystop, scale in parameters:
        out_img, box_list, bbList3 = find_cars(C_frames, ystart, ystop, scale, svc, X_scaler, orient,
                                pix_per_cell, cell_per_block, spatial_size, hist_bins, draw_img)
        draw_img = out_img
        boxes.append(box_list)

        
    # Accumulator to smooth
    auxBox = boxes
    if not math.fmod(count,10):
        auxBox[0:int(len(boxes)*0.35)] = []
        releaseBox = auxBox
           
    
    # Detecting Heat labels
    heat = add_heat(heat,releaseBox)
    heat = apply_threshold(heat,3)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)   
    draw_labeled, Labeled_bboxList  = draw_labeled_bboxes(np.copy(C_frames), labels)
    
    bb = np.array(Labeled_bboxList)   
    bb_NMS = non_max_suppression_fast(bb, 0.3)
    
    # Save Frame by Frame. Debug Pourpose
    draw_bboxes(C_frames_B, bb_NMS, (0,0,220) ) 

    path = './temp/' + '{:05d}'.format(count) + 'frames' + '.png'
    cv2.imwrite(path,C_frames_B)

    count += 1
    
temp = glob.glob('temp/*.png')
clip = ImageSequenceClip(temp, fps=24)
clip.write_videofile('finalisimo000.mp4')

#video.write_videofile('video.mp4', audio=False)
