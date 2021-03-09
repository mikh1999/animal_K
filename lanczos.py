import numpy as np
from PIL import Image
import cv2

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline


# array = np.random.randint(0, 1300, size=(10, 256, 256))
# newarray = np.zeros((10, 224, 224))
# newarray[0] = cv2.resize(array[0], dsize=(224, 224), interpolation=cv2.INTER_LANCZOS4)
# print(newarray)

imgorig = cv2.imread('/home/mikh1999/PycharmProjects/Animal_K/pytorch-ZSSR/test_data/fgbg000011.jpg')
imgorig2 = cv2.imread('/home/mikh1999/PycharmProjects/Animal_K/pytorch-ZSSR/test_data/fgbg000024.png')
imgorig3 = cv2.imread('/home/mikh1999/PycharmProjects/Animal_K/pytorch-ZSSR/test_data/fgbg000094.png')

img = cv2.cvtColor(imgorig, cv2.COLOR_BGR2RGB)
# img = cv2.cvtColor(imgorig, cv2.COLOR_RGB2BGR)



img2 = cv2.cvtColor(imgorig2, cv2.COLOR_BGR2RGB)
print(img2.shape)

img3 = cv2.cvtColor(imgorig3, cv2.COLOR_BGR2RGB)
print(img3.shape)
interpolation_algorithm = [
    # ("nearest", cv2.INTER_NEAREST),
    # ("bilinear", cv2.INTER_LINEAR),
    # ("bicubic", cv2.INTER_CUBIC),
    # ("area", cv2.INTER_AREA),
    ("lanczos4", cv2.INTER_LANCZOS4)
]


def resize_test(img, factor, is_plot=True, file_name=None):
    height, width, channels = img.shape
    height2, width2 = int(height * factor), int(width * factor)
    print('orig size:', height, width)
    print('resize size:', height2, width2)

    imgs = []
    # for alg in interpolation_algorithm:
    #     img_r = cv2.resize(img, (width2, height2), interpolation=alg[1])
    #     imgs.append(img_r)

    img_r = cv2.resize(img, (width2, height2), interpolation=cv2.INTER_LANCZOS4)
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    # img_r = cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR)
    cv2.imwrite('11.png', img_r)
    print(img.shape)
    plt.imshow(img_r)
    plt.show()

    # if is_plot:
    #     plt.figure(figsize=(11, 2))
    #     plt.subplot(1, len(imgs) + 1, 1)
    #     plt.title('orig')
    #     plt.imshow(img)
    #     plt.axis('off')
    #     for i in range(len(imgs)):
    #         plt.subplot(1, len(imgs) + 1, i + 2)
    #         plt.title(interpolation_algorithm[i][0])
    #         plt.imshow(imgs[i])
    #         plt.axis('off')
    #     plt.subplots_adjust(wspace=0)
    #     if file_name != None:
    #         plt.savefig(file_name)
    #     plt.show()

    return imgs



print('UP')
img_up0 = resize_test(img, 8, file_name='opencv_resize_up1.png')
# img_up1 = resize_test(img2, 2, file_name='opencv_resize_up2.png')
# img_up2 = resize_test(img3, 2, file_name='opencv_resize_up3.png')