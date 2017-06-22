#!/usr/bin/env python

from skimage.io import imread
from skimage import img_as_float
import pylab
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def recreate_image(x_true, codebook, labels, w, h, how):
	#Recreate the (compressed) image from the code book & labels
	cb = np.zeros(codebook.shape)
	if how == 'mean':
		for i in range(labels.max()+1):
			cb[i] = x_true[np.where(labels == i)].mean(axis=0)
	if how == 'median':
		for i in range(labels.max()+1):
			cb[i] = np.median(x_true[np.where(labels == i)], axis=0)
	
	d = codebook.shape[1]
	image = np.zeros((w, h, d))
	label_idx = 0
	for i in range(w):
		for j in range(h):
			image[i][j] = cb[labels[label_idx]]
			label_idx += 1
	print '\nrecreate output\n', cb
	return image


def PSNR(x_true, x_pred):
	x_max = np.amax(x_true)
	mse = (np.square(x_true - x_pred)).sum() / np.prod(x_true.shape)
	#print 'PSNR output\nx_max = %0.3f\nprod = %d\nmse = %0.3f\n' % (x_max, np.prod(x_true.shape), mse)
	return 10*np.log10(x_max*x_max/mse)


if __name__ == '__main__':
	image = img_as_float(imread('parrots.jpg'))

	w, h, d = original_shape = tuple(image.shape)

	X = np.reshape(image, (w * h, d))

	for nc in range(1,20+1):
		#print 'Training model'
		kmeans = KMeans(n_clusters = nc, init='k-means++', random_state=241).fit(X)

		#print 'Predicting color indices on the full image (k-means)'
		labels = kmeans.predict(X)

		#print 'Recreating image'
		X_pred_mean = recreate_image(X, kmeans.cluster_centers_, labels, w, h, 'mean')
		psnr_mean = PSNR(image, X_pred_mean)
		print '\n#clusters = %d, PSNR = %0.3f (mean)' % (nc, psnr_mean)

		X_pred_median = recreate_image(X, kmeans.cluster_centers_, labels, w, h, 'median')
		psnr_median = PSNR(image, X_pred_median)
		print '#clusters = %d, PSNR = %0.3f (median)' % (nc, psnr_median)

		if (max([psnr_mean, psnr_median]) > 20):
			print 'It takes min %d clusters to go over peak signal-to-noise ratio 20 dB\n' % nc
			break
