import cv2
import math
import numpy as np
from images2gif import writeGif
def show_image(image, image_name='Houxianxu', resize=False):
    if resize:
        image = cv2.resize(image, (resize, resize))
    cv2.imshow(image_name, image)
    key = cv2.waitKey(0)
    if key == 27: # ESC
        cv2.destroyAllWindows()

def get_image_list(image):
    """
    From a long image to sub small images
    """
    height = image.shape[0]
    width = image.shape[1]
    image_list = []
    num_image = width / height
    for i in xrange(num_image):
        image_list.append(image[:, i*height: (i+1)*height, :])
        # show_image(image_list[i])
    return image_list


def build_gif_one_image(image_file):
    output_file = image_file.split('.')[0] + '.gif'
    image = cv2.imread(image_file)[:,:, ::-1]  # from BGR->RGB
    subimages = get_image_list(image)
    writeGif(output_file, subimages, 0.1)


def build_gif_multi_images(image_file_list, output_file='combined.gif', annotations=None):
    image_list = []  # list of list images
    for image_file in image_file_list:
        image = cv2.imread(image_file)[:, :, ::-1]
        image_list.append(get_image_list(image))

    if annotations is not None:
        ann_image_list = []
        for image_file in annotations:
            image = cv2.imread(image_file)[:, :, ::-1]
            ann_image_list.append(image)

    # output size
    sub_size = image_list[0][0].shape[0]
    num_small_image = len(image_list[0])
    output_size = int(math.sqrt(len(image_list))) # number of small images in a row
    output_image_list = []
    for i in xrange(num_small_image):
        if annotations is not None:
            output_image = np.zeros((output_size * sub_size, (output_size+1) * sub_size, 3), dtype=np.uint8) # here should use dtype
        else:
            output_image = np.zeros((output_size * sub_size, output_size * sub_size, 3), dtype=np.uint8) # here should use dtype
        j = 0
        for m in xrange(output_size):
            for n in xrange(output_size):
                output_image[m*sub_size:(m+1)*sub_size, n*sub_size:(n+1)*sub_size, :] = image_list[j][i]
                cv2.imwrite('output_image.jpg', output_image[m*sub_size:(m+1)*sub_size, n*sub_size:(n+1)*sub_size, :])
                j += 1
            if annotations is not None:
                output_image[m*sub_size:(m+1)*sub_size, (n+1)*sub_size:(n+2)*sub_size, :] = ann_image_list[m]

        output_image_list.append(output_image)

    writeGif(output_file, output_image_list, 0.12)

def main():
    # image_file = 'linear_011116.jpg'
    # build_gif_one_image(image_file)
    # each image is a long image with multi subimages
    image_file_list = ['non_smiling_linear_001290.jpg','non_smiling_linear_007113.jpg','non_smiling_linear_011116.jpg','non_smiling_linear_011190.jpg',
                        'smiling_linear_000632.jpg','smiling_linear_011125.jpg','smiling_linear_011149.jpg','smiling_linear_011141.jpg',
                        'non_glass_linear_001290.jpg','non_glass_linear_011116.jpg','non_glass_linear_011121.jpg','non_glass_linear_011190.jpg',
                        'glass_linear_004465.jpg','glass_linear_011146.jpg','glass_linear_012958.jpg','glass_linear_032439.jpg']
    annotations = ['non-smiling.jpg' ,'smiling.jpg', 'non-eyeglass.jpg', 'eyeglass.jpg']

    # build_gif_multi_images(image_file_list, annotations=annotations)

    random_image_list = ['linear_walk_random' + str(i) + '.jpg' for i in xrange(1, 26)]
    build_gif_multi_images(random_image_list, 'random.gif')

if __name__ == '__main__':
    main()


