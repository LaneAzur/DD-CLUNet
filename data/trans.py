import collections
import numpy as np
import random
import cv2 as cv



class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False): # class -> func()
        # image: nhwtc
        # shape: no first dim
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            # how to know  if the last dim is channel??
            # nhwtc vs nhwt??
            shape = im.shape[1:dim+1]
            # print(dim,shape) # 3, (240,240,155)
            self.sample(*shape)

        if isinstance(img, collections.abc.Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)] # img:k=0,label:k=1

        return self.tf(img)

    def __str__(self):
        return 'Identity()'

Identity = Base

class RandomFlip(Base):
    def __init__(self, axis=1):
        self.axis = axis

    def __call__(self, image_list):
        do_flip = np.random.random(1)
        if do_flip > 0.5:
            for i in range(len(image_list)):
                image_list[i] = np.flip(image_list[i], axis=self.axis)
        return image_list

# class RandomFlip(Base):
#     # mirror flip across all x,y,z
#     def __init__(self,axis=0):
#         # assert axis == (1,2,3) # For both data and label, it has to specify the axis.
#         self.axis = (1,2,3)
#         self.x_buffer = None
#         self.y_buffer = None
#         self.z_buffer = None
#
#     def sample(self, *shape):
#         self.x_buffer = np.random.choice([True,False])
#         self.y_buffer = np.random.choice([True,False])
#         self.z_buffer = np.random.choice([True,False])
#         return list(shape) # the shape is not changed
#
#     def tf(self,img,k=0): # img shape is (1, 240, 240, 155, 4)
#         if self.x_buffer:
#             img = np.flip(img,axis=self.axis[0])
#         if self.y_buffer:
#             img = np.flip(img,axis=self.axis[1])
#         if self.z_buffer:
#             img = np.flip(img,axis=self.axis[2])
#         return img


class Seg_norm(Base):
    def __init__(self, ):
        a = None
        self.seg_table = np.array([0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255])
    def tf(self, img, k=0):
        if k == 0:
            return img
        img_out = np.zeros_like(img)
        for i in range(len(self.seg_table)):
            img_out[img == self.seg_table[i]] = i
        return img_out

class CenterCrop(object):
    def __init__(self, size_ratio):
        if isinstance(size_ratio, list) or isinstance(size_ratio, tuple):
            self.size_ratio_x, self.size_ratio_y, self.size_ratio_z = size_ratio
        else:
            raise ValueError

    def __call__(self, image_list):
        x, y, z = image_list[0].shape

        self.size_x = self.size_ratio_x
        self.size_y = self.size_ratio_y
        self.size_z = self.size_ratio_z

        if x < self.size_x or y < self.size_y and z < self.size_z:
            raise ValueError

        x1 = int((x - self.size_x) / 2)
        y1 = int((y - self.size_y) / 2)
        z1 = int((z - self.size_z) / 2)

        # for i in range(len(image_list)):
        #
        #     image_list[i] = image_list[i][x1: x1 + self.size_x, y1: y1 + self.size_y, z1: z1 + self.size_z]

        image = image_list[0][x1: x1 + self.size_x, y1: y1 + self.size_y, z1: z1 + self.size_z]

        image_list = [image]
        return image_list

class RandomCrop(Base):
    def __init__(self, size):
        if isinstance(size, list) or isinstance(size, tuple):
            self.size_x, self.size_y, self.size_z = size
        else:
            raise ValueError

    def __call__(self, image_list):

        x, y, z = image_list[0].shape
        if x < self.size_x or y < self.size_y and z < self.size_z:
            raise ValueError

        x1 = random.randint(0, x - self.size_x)
        y1 = random.randint(0, y - self.size_y)
        z1 = random.randint(0, z - self.size_z)


        image = image_list[0][x1: x1 + self.size_x, y1: y1 + self.size_y, z1: z1 + self.size_z]
        # image = image_list[1][x1: x1 + self.size_x, y1: y1 + self.size_y, z1: z1 + self.size_z]

        # for i in range(len(image_list)):
        #     print(i)
        #     image_list[i] = image_list[i][x1: x1 + self.size_x, y1: y1 + self.size_y, z1: z1 + self.size_z]
        #     print('image_list',image_list[i].shape)
        image_list = [image]
        return image_list

class RandomRotate(Base):
    def __init__(self):
        self.angle_list = [-10, -5, 0, 5, 10]

    def __call__(self, image_list):
        angle = self.angle_list[random.randint(0, 4)]

        img_slice = image_list[0]
        raws, z, cols = img_slice.shape
        M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (raws - 1) / 2.0), angle, 1)

        for i in range(len(image_list)):
            # print("rotate", key, i)
            for j in range(z):
                image_list[i][:, j, :] = self.rotate(image_list[i][:, j, :], M, raws, cols, is_target=(i==3))
        return image_list

    def rotate(self, img_slice, M, raws, cols, is_target=False):
        if not is_target:
            img_rotated = cv.warpAffine(img_slice, M, (cols, raws))
        else:
            img_rotated = cv.warpAffine(img_slice, M, (cols, raws), flags=cv.INTER_NEAREST)

        return img_rotated

class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)

