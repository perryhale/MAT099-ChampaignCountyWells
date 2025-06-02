import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from library.data import crop_matrix_crs


### parameters

# hydraulic conductivity map
K_BOUND_N = 49.0000
K_BOUND_S = 24.0000 # long
K_BOUND_W = -126.0000
K_BOUND_E = -66.0000 # lat

# safety switch (off)
Image.MAX_IMAGE_PIXELS = 1e16 # for 100m data


### main

# load data
k = np.array(Image.open(sys.argv[1]))
data_location = pd.read_csv('data/champaign_county_wells/OB_LOCATIONS.csv')
data_surface = pd.read_csv('data/processed/data_surface.csv', index_col='TIMESTAMP')
data_surface.columns = data_surface.columns.astype('int32')

# determine well and bounding area coordinates
data_coords = data_location.set_index('P_NUMBER').loc[data_surface.columns, ['LONG_NAD_83', 'LAT_NAD_83']].to_numpy()
data_bound_n = data_coords[:,1].max()
data_bound_s = data_coords[:,1].min()
data_bound_w = data_coords[:,0].min()
data_bound_e = data_coords[:,0].max()
print("Bounding area:")
print("North:", data_bound_n)
print("South:", data_bound_s)
print("West:", data_bound_w)
print("East:", data_bound_e)
# North: -87.981028
# South: -88.463237
# West: 40.05338
# East: 40.385156

# crop k and unpack
k_cropped, target_idx, _ = crop_matrix_crs(k, (K_BOUND_N, K_BOUND_S, K_BOUND_W, K_BOUND_E), (data_bound_n, data_bound_s, data_bound_w, data_bound_e))
target_north_idx, target_south_idx, target_west_idx, target_east_idx = target_idx

# interactive plot
plt.imshow(k, vmin=0, vmax=25)
plt.colorbar()
plt.scatter([target_west_idx, target_east_idx], [target_north_idx, target_south_idx], c='red', marker='+')
plt.gca().add_patch(Rectangle((target_west_idx, target_north_idx), k_cropped.shape[1], k_cropped.shape[0],
	edgecolor='red',
	facecolor=None,
	fill=False,
	lw=1
))
plt.tight_layout()
plt.show()
