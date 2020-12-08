from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import animation
import os

data = np.loadtxt("Field_map.txt")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = data[:,0]
Y = data[:,1]
Z = -data[:,2] #Camera is set to Z position 0
C = data[:,3]

img = ax.scatter(X, Y, Z, c=C, cmap=plt.hot())
fig.colorbar(img)

# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
   ax.plot([xb], [yb], [zb], 'w')

#Lets draw some animation and save it as gif
def rotate(angle):
    ax.view_init(azim=angle)
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
rot_animation.save(os.path.join(os.getcwd(),'rotation.gif'), dpi=80, writer='Pillow')


plt.grid()
plt.show()