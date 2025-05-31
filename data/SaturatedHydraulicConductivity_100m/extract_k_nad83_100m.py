import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = 1e16

k = np.array(Image.open('./KSat_Arithmetic_100m.tif'))
k_width = k.shape[1]
k_height = k.shape[0]
print(k.shape)

# bounding
westbc = -126.0000
eastbc = -66.0000
northbc = 49.0000
southbc = 24.0000

dx = (eastbc - westbc)/k_width
dy = (northbc-southbc)/k_height

print(dx)
print(dy)

# target
xmin = -88.333474
xmax = -88.288459
ymin = 40.116967
ymax = 40.223612

xmin_coord = int((xmin-westbc) / dx)
xmax_coord = int((xmax-westbc) / dx)
ymin_coord = int((northbc-ymin) / dy)
ymax_coord = int((northbc-ymax) / dy)

print('nw', xmin_coord, ymax_coord)
print('se', xmax_coord, ymin_coord)

# (32315, 59991)
# 0.0010001500225033755
# 0.0007736345350456444
# nw 37660 11344
# se 37705 11482

crop_x = xmin_coord
crop_y = ymax_coord
crop_w = xmax_coord-xmin_coord
crop_h = ymin_coord-ymax_coord

k_cropped = k[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

np.save(__file__.replace('.py',''), k_cropped)

plt.imshow(k_cropped)
plt.colorbar()
plt.tight_layout()
plt.show()
