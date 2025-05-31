import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

k = np.array(Image.open('./KSat_Arithmetic_1km.tif'))
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

# (3232, 6000)
# 0.01
# 0.007735148514851485
# nw 3766 1134
# se 3771 1148

crop_x = 3766
crop_y = 1134
crop_w = 3771-3766
crop_h = 1148-1134

plt.imshow(k[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w])
plt.colorbar()
plt.tight_layout()
plt.show()
