import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

tif = Image.open(sys.argv[1])
k = np.array(tif)
print(k.shape)

VMIN = 0
VMAX = 10

plt.imshow(k, vmin=VMIN, vmax=VMAX)
plt.colorbar()
plt.tight_layout()
plt.show()
