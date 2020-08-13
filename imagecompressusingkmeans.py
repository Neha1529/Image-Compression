
from skimage import io
from sklearn.cluster import KMeans
import numpy as np

from google.colab import files
from IPython.display import Image

uploaded = files.upload()

Image('dog_PNG50321.png')

image = io.imread('dog_PNG50321.png')
io.imshow(image)
io.show()

rows = image.shape[0]
cols = image.shape[1]

image.shape

image = image.reshape(image.shape[0]*image.shape[1],4)
kmeans = KMeans(n_clusters= 128, n_init= 10 ,max_iter = 200)
kmeans.fit(image)

clusters = np.asarray(kmeans.cluster_centers_ , dtype = np.uint8)
labels = np.asarray(kmeans.labels_, dtype= np.uint8)
labels = labels.reshape(rows,cols)

io.imsave('compressed_image.png',labels)
np.save('codebook_dog.npy' , clusters)

centres = np.load('codebook_dog.npy')
c_image = io.imread('compressed_image.png')

image = np.zeros((c_image.shape[0],c_image.shape[1],4),dtype = np.uint8)
for i in range(c_image.shape[0]):
  for j in range(c_image.shape[1]):
    image[i,j,:] = centres[c_image[i,j],:]

io.imsave('reconstructed_dog.png',image)
io.imshow(image)
io.show()


