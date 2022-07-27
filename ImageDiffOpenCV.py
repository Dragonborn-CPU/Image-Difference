import cv2

''' Program by: Ethan S
Program that compares two images and calculates similarity while also matching common points between the two images.
'''
original = cv2.imread('C:/Users/admin/Downloads/DSC00067.JPG')  # 1st Image
image_to_compare = cv2.imread('C:/Users/admin/Downloads/DSC00068.JPG')  # 2nd Image

if original.shape == image_to_compare.shape:
    print("The images have same size and channels")
    difference = cv2.subtract(original, image_to_compare)
    b, g, r = cv2.split(difference)

    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("The images are COMPLETELY equal")
    else:
        print("The images are NOT equal")

sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(desc_1, desc_2, k=2)

good_points = []
for m, n in matches:
    if m.distance < .8*n.distance:  # change here to affect accuracy
        good_points.append(m)

number_keypoints = 0
if len(kp_1) <= len(kp_2):
    number_keypoints = len(kp_1)   # may want to switch kp_2 and kp_1 here
else:
    number_keypoints = len(kp_2)

print("Keypoints 1st Image: " + str(len(kp_1)))
print("Keypoints 2nd Image: " + str(len(kp_2)))
print("Matches:", len(good_points))
print("How good is the match: ", len(good_points) / number_keypoints * 100)

result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)


cv2.imshow("Comparison", cv2.resize(result, None, fx=0.25, fy=0.25))
cv2.imwrite("feature_matching.jpg", result)  # save image
cv2.imshow("1st Image", cv2.resize(original, None, fx=0.25, fy=0.25))
cv2.imshow("2nd Image", cv2.resize(image_to_compare, None, fx=0.25, fy=0.25))
cv2.waitKey(0)
cv2.destroyAllWindows()
