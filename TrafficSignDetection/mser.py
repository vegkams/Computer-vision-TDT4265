import cv2
import numpy as np
# Params for MSER
_delta = 4
_min_area = 50
_max_area = 8000
_max_variation = 0.1
_min_diversity = 5
_max_evolution = 1
_area_threshold = 0.5
_min_margin = 0.1
_edge_blur_size = 1

#Params for sorting areas
_width_min = 14
_width_max = 100
_height_min = 14
_height_max = 110
_hw_min = 0.8
_hw_max = 1.2

img = cv2.imread("00001.ppm");
vis = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mser = cv2.MSER_create(_delta, _min_area, _max_area, _max_variation, _min_diversity, \
                       _max_evolution, _area_threshold, _min_margin , _edge_blur_size)
# bbox: [min x, min y, max x-min x, max y-min y]
regions, bbox = mser.detectRegions(gray)
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
        filtered_regions.append(regions[i])
        filtered_bbox.append(box)
        i += 1
i = 0
print(filtered_regions[0])
'''filtered_regions2 = list()
filtered_bbox2 = list()
for region in filtered_regions:
    peri = cv2.arcLength(region, True)
    approx = cv2.approxPolyDP(region, 0.04 * peri, True)
    print(approx)
    print("mellomrom \n")
    if len(approx) == 3:
        filtered_regions2.append(region)
        filtered_bbox2.append(filtered_bbox[i])
        print("Found triangle")
    elif len(approx) == 4:
        filtered_regions2.append(region)
        filtered_bbox2.append(filtered_bbox[i])
        print("Found square")
    i += 1
'''

print(len(filtered_bbox))
print(len(filtered_regions))
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in filtered_regions]
#print(hulls)
cv2.polylines(vis, hulls, 1, (0, 255, 0))
cv2.imshow('img', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()