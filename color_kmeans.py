# USAGE
# python color_kmeans.py --image images/jp.png --clusters 3

# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
import utils
import cv2
from colormap import hex2rgb

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.style.use('dark_background')

# show our image
plt.figure(figsize=(4, 4))
plt.axis("off")
plt.imshow(image)

# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))

# cluster the pixel intensities
clt = KMeans(n_clusters = args["clusters"])
clt.fit(image)

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = utils.centroid_histogram(clt)
# bar = utils.plot_colors(hist, clt.cluster_centers_)

primaryColor, secondaryColor = utils.findPri_Sec(hist, clt.cluster_centers_)

print(f'primary = {primaryColor}, secondary = {secondaryColor}')

# show our color bar
# plt.figure()
# plt.axis("off")

# plt.imshow(bar)
# plt.style.use('dark_background')

plt.figure(figsize=(4, 4))
currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((0.2 - 0.1, 0.5 - 0.1), 0.2, 0.2, fill = True, color = primaryColor))
currentAxis.add_patch(Rectangle((0.5 - 0.1, 0.5 - 0.1), 0.2, 0.2, fill = True, color = secondaryColor))
plt.show()