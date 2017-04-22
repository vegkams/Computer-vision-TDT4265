import cv2
import numpy as np
# Params for MSER
_delta = 2
_min_area = 100
_max_area = 10000
_max_variation = 0.25
_min_diversity = 0.2
_max_evolution = 150
_area_threshold = 1.01
_min_margin = 0.1
_edge_blur_size = 1

#Params for sorting areas
_width_min = 14
_width_max = 150
_height_min = 14
_height_max = 150
_hw_min = 0.85
_hw_max = 1.15

img = cv2.imread("datasets/TrainIJCNN2013/00012.ppm");
vis = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#mser = cv2.MSER_create()
mser = cv2.MSER_create(_delta, _min_area, _max_area, _max_variation, _min_diversity, \
                       _max_evolution, _area_threshold, _min_margin , _edge_blur_size)
# bbox: [min x, min y, max x-min x, max y-min y]
regions, bbox = mser.detectRegions(img)
filtered_regions = list()
filtered_bbox = list()
i = 0
# Remove non-suitable boxes
for box in bbox:
    if (box[3]/box[2] > _hw_max) or (box[3]/box[2] < _hw_min) or (box[2] > _width_max) \
            or (box[2] < _width_min) or (box[3] > _height_max) or (box[3] < _height_min):
        i += 1
        pass
    else:
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3]
        filtered_regions.append(regions[i])
        filtered_bbox.append(box)
        i += 1

print(filtered_regions[0])

# Duplicate each box, such that each box has at least one overlapping box
filtered_bbox_dup = filtered_bbox.copy()
print(len(filtered_bbox))
for box in filtered_bbox_dup:
    filtered_bbox.append(box)

rectList, weights = cv2.groupRectangles(filtered_bbox, 1, 0.05)
rectList_dup = rectList.tolist()
rectList = rectList.tolist()
for box in rectList_dup:
    rectList.append(box)


hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in filtered_regions]
#print(hulls)
#cv2.polylines(vis, hulls, 1, (0, 255, 0))
#for box in filtered_bbox:
#    cv2.rectangle(vis,(box[0], box[1]),(box[2], box[3]),(0, 0, 255), 1)

for rectangle in rectList:
    cv2.rectangle(vis,(rectangle[0], rectangle[1]),(rectangle[2], rectangle[3]),(255, 0, 255), 2)
cv2.imshow('img', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()