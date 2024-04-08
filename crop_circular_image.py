import cv2
import numpy as np

# read image
img = cv2.imread('./images/lena.jpg')
hh, ww = img.shape[:2]
hh2 = hh // 2 # around down
ww2 = ww // 2 # around down

# define circles
radius1 = 25
radius2 = 75
xc = hh // 2
yc = ww // 2

# draw filled circles in white on black background as masks
mask1 = np.zeros_like(img)
mask1 = cv2.circle(mask1, (xc, yc), radius1, (255, 255, 255), -1)
mask2 = np.zeros_like(img)
mask2 = cv2.circle(mask2, (xc, yc), radius2, (255, 255, 255), -1)

# subtract masks and make into single channel
mask = cv2.subtract(mask2, mask1)

# put mask into alpha chanel of input
result  = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
result[:, :, 3] = mask[:, :, 0]

# save results
cv2.imwrite('lena_mask1.png', mask1)
cv2.imwrite('lena_mask2.png', mask2)
cv2.imwrite('lena_masks.png', mask)
cv2.imwrite('lena_circle_masks.png', result)

cv2.imshow('image', img)
cv2.imshow('mask1', mask1)
cv2.imshow('mask2', mask2)
cv2.imshow('mask', mask)
cv2.imshow('masked image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

