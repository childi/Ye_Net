import numpy as np


def fun(numstr: str):
    """需要输入为字符串"""
    num = numstr.upper()
    if 'E' not in num:
        return numstr

    e = num.find('E')

    big = num[:e]

    big = float(big)
    # print(big)
    tmp = num[e+1:]
    tmp = int(tmp)
    # print(tmp)
    num_tmp = big*pow(10, tmp)
    return num_tmp


def process_pred(line, file):
    """Compares the given object to the current ObjectLabel instance.

    :param line
    :return: object(not truncation,occlusion, alpha,difficult,iou)
    """
    obj = ObjectLabel()
    s = line.split(' ')
    file = file.split('.')
    obj.type = s[0]
    obj.truncation = 0.
    obj.occlusion = 0.
    obj.alpha = 0.
    obj.x1 = float(s[4])
    obj.y1 = float(s[5])
    obj.x2 = float(s[6])
    obj.y2 = float(s[7])
    obj.h = float(s[8])
    obj.w = float(s[9])
    obj.l = float(s[10])
    obj.t = (float(s[11]), float(s[12]), float(s[13]))
    obj.ry = float(s[14])
    obj.score = float(s[15])
    obj.difficulty = -1
    obj.iou = 0.
    obj.img_cnt = int(file[0])

    return obj


def map_process(obj, cam_matrix):
    ratio_dict = {0: 128., 1: 64., 2: 32., 3: 16., 4: 8.}
    C_u = cam_matrix[0][2]
    delta_u = cam_matrix[0][3]
    C_v = cam_matrix[1][2]
    delta_v = cam_matrix[1][3]
    f_u = cam_matrix[0][0]
    f_v = cam_matrix[1][1]
    delta_z = cam_matrix[2][3]
    C_u_0 = round(C_u)
    C_v_0 = round(C_v)
    x_0 = (delta_u - C_u * delta_z) / f_u
    y_0 = (delta_v - C_v * delta_z) / f_v
    z_0 = delta_z
    u_0 = -C_u_0
    v_0 = -C_v_0
    epsilon_u_0 = -(C_u - C_u_0) / f_u
    epsilon_v_0 = -(C_v - C_v_0) / f_v
    dh = round(1 * 128 - (C_v_0 - 64))
    dw = round(5 * 128 - (C_u_0 - 64))

    x = obj.t[0] + x_0
    y = obj.t[1] + y_0
    z = obj.t[2] + z_0

    box_distance_bev = (x ** 2 + z ** 2) ** (1 / 2)
    if box_distance_bev < 3.5:
        theta = np.arctan(x / z)
        obj.alpha = obj.ry + theta
        obj.feature_map = -1
        obj.x_map = -1
        return obj
    feature_map_group = int(np.floor(np.log2(box_distance_bev / 3.5)))

    box_u = (x / z) * f_u - u_0 + dw
    ratio = ratio_dict[feature_map_group]
    x_map = int(np.floor(box_u / ratio))

    theta = np.arctan(x / z)
    obj.alpha = obj.ry + theta

    obj.feature_map = feature_map_group
    obj.x_map = x_map

    return obj


def difficulty_process(obj):
    if obj.truncation <= 0.15 and obj.occlusion <= 0 and (obj.y2 - obj.y1) >= 40:
        obj.difficulty = 0.
    elif obj.truncation <= 0.30 and obj.occlusion <= 1 and (obj.y2 - obj.y1) >= 25:
        obj.difficulty = 1.
    elif obj.truncation <= 0.50 and obj.occlusion <= 2 and (obj.y2 - obj.y1) >= 25:
        obj.difficulty = 2.
    else:
        obj.difficulty = 3.

    return obj


def process_gt(line, file):
    """Compares the given object to the current ObjectLabel instance.

    :param line
    :return: object(not  alpha,difficult,iou,score)
    """
    obj = ObjectLabel()
    s = line.split(' ')
    file = file.split('.')
    obj.type = s[0]
    obj.truncation = float(s[1])
    obj.occlusion = float(s[2])
    obj.alpha = 0.
    obj.x1 = float(s[4])
    obj.y1 = float(s[5])
    obj.x2 = float(s[6])
    obj.y2 = float(s[7])
    obj.h = float(s[8])
    obj.w = float(s[9])
    obj.l = float(s[10])
    obj.t = (float(s[11]), float(s[12]), float(s[13]))
    obj.ry = float(s[14])
    obj.score = 0.
    obj.difficulty = 4.
    obj.iou = 0.
    obj.img_cnt = int(file[0])

    return obj

def get_p(p_path):
    with open(p_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line[0:2] == 'P2':
                # print(line.split(' '))
                line = line.split(' ')
                line = line[1:]
                # print(line)
                for tmp in range(12):
                    line[tmp] = float(fun(line[tmp]))
                p = np.reshape(line, [3, 4])
    return p

class ObjectLabel:
    """Object Label Class
    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                      'Misc' or 'DontCare'

    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                      truncated refers to the object leaving image boundaries

    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                      0 = fully visible, 1 = partly occluded
                      2 = largely occluded, 3 = unknown

    1    alpha        Observation angle of object, ranging [-pi..pi]

    4    bbox         2D bounding box of object in the image (0-based index):
                      contains left, top, right, bottom pixel coordinates

    3    dimensions   3D object dimensions: height, width, length (in meters)

    3    location     3D object location x,y,z in camera coordinates (in meters)

    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]

    1    score        Only for results: Float, indicating confidence in
                      detection, needed for p/r curves, higher is better.
    """

    def __init__(self):
        # obj = ObjectLabel()
        self.type = ""  # Type of object
        self.truncation = 0.
        self.occlusion = 0.
        self.alpha = 0.
        self.x1 = 0.
        self.y1 = 0.
        self.x2 = 0.
        self.y2 = 0.
        self.h = 0.
        self.w = 0.
        self.l = 0.
        self.t = (0., 0., 0.)
        self.ry = 0.
        self.score = 0.
        self.difficulty = 0.
        self.iou = 0.
        self.img_cnt = 0.
        self.feature_map = 0.
        self.x_map = 0.
        self.match = 0.
        self.match_obj = None

    def __eq__(self, other):
        """Compares the given object to the current ObjectLabel instance.

        :param other: object to compare to this instance against
        :return: True, if other and current instance is the same
        """
        if not isinstance(other, ObjectLabel):
            return False

        if self.__dict__ != other.__dict__:
            return False
        else:
            return True

