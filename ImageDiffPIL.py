from PIL import Image, ImageChops

''' Program by: Ethan S
Super simple program to find difference between two images through image subtraction.
'''
image_1 = Image.open('C:/Users/admin/Downloads/DSC00057.JPG')
image_2 = Image.open('C:/Users/admin/Downloads/DSC00058.JPG')

Difference_0 = ImageChops.difference(image_1, image_2)
Difference_0.show()