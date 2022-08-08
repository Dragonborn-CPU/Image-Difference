from skimage.metrics import structural_similarity
import cv2
import numpy as np


def resize():
    global img, img2
    res = .3
    w, h = int(img.shape[1] * res), int(img.shape[0] * res)
    img = cv2.resize(img, (w, h))
    img2 = cv2.resize(img2, (w, h))


def RemoveNoise(src, src2):
    global dst, dst2
    dst = cv2.fastNlMeansDenoisingColored(src, None, 11, 6, 7, 21)
    dst2 = cv2.fastNlMeansDenoisingColored(src2, None, 11, 6, 7, 21)


def ImageDiff(dst, dst2):
    # Load images
    before = dst
    after = dst2

    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image Similarity: {:.4f}%".format(score * 100))

    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    # threshold image
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()

    # draw contours + bounding box
    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:  # set limits for contours
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36, 255, 12), 2)
            cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)
            cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

    cv2.imshow('first image', before)
    cv2.imshow('second image', after)
    cv2.imshow('diff', diff)
    cv2.imshow('diff_box', diff_box)
    # cv2.imshow('mask', mask)
    # cv2.imshow('filled after', filled_after)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = cv2.imread('C:/Users/admin/Downloads/DSC00002.JPG')
    img2 = cv2.imread('C:/Users/admin/Downloads/DSC00003.JPG')
    resize()
    RemoveNoise(img, img2)
    ImageDiff(dst, dst2)
