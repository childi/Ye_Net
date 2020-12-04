##############################
# for yolo3d
##############################
import os
import argparse

# import train_val


def convert_voc_annotation(data_path, data_type, anno_path):

    # classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
    classes = ['Car']  # , 'Pedestrian', 'Cyclist'
    img_inds_file = os.path.join(data_path, data_type + '.txt')  #
    data_path = os.path.join(data_path, 'training')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]

    with open(anno_path, 'a') as f:
        for image_ind in image_inds:
            # print(image_ind)
            image_path = os.path.join(data_path, 'image_2', image_ind + '.png')
            annotation = image_path
            label_path = os.path.join(data_path, 'label_2', image_ind + '.txt')
            with open(label_path, 'r') as la:
                txt = la.readlines()
                label_file = [line.strip() for line in txt]
            for label in label_file:
                label = label.split()
                if label[0] not in classes:
                    continue
                box3d = label[8:14]
                class_ind = classes.index(label[0].strip())
                h = str(float(box3d[0]))
                w = str(float(box3d[1]))
                l = str(float(box3d[2]))
                x = str(float(box3d[3]))
                y = str(float(box3d[4]))
                z = str(float(box3d[5]))
                ry = label[14]
                annotation += ' ' + ','.join([y, x, z, l, w, h, ry, str(class_ind)])
            print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_path", default="/media/dataset/Kitti/object/")
#     parser.add_argument("--train_annotation", default="/home/zhangy/yolov3/data/dataset/kitti_train3d.txt")
#     parser.add_argument("--test_annotation",  default="/home/zhangy/yolov3/data/dataset/kitti_test3d.txt")
#     flags = parser.parse_args()
#
#     if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
#     if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)
#
#     # TODO:split train/val
#     # with open('trainval.txt', 'r') as f:
#     #     txt = f.readlines()
#     #     train_num = round(0.9 * len(txt))
#     # train_val.split(train_num)
#
#     num1 = convert_voc_annotation(flags.data_path, 'train', flags.train_annotation)
#     num2 = convert_voc_annotation(flags.data_path, 'val', flags.test_annotation)
#     print('=> The number of image for train is: %d\tThe number of image for test is:%d' % (num1, num2))

###########################
# convert F-P'prim 2dbox into the form of yolov3' pred 2dbox
###########################
# import os
# import argparse
# import shutil
#
#
# def convert_voc_annotation(data_path, data_type, anno_path, out_path):
#
#     # classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
#     classes = ['Car']  # , 'Pedestrian', 'Cyclist'
#     img_inds_file = os.path.join(data_path, data_type + '.txt')  #
#     # data_path = os.path.join(data_path, 'training')
#     with open(img_inds_file, 'r') as f:
#         txt = f.readlines()
#         image_inds = [line.strip() for line in txt]
#
#     with open(anno_path, 'r') as f:
#         txt = f.readlines()
#         prim = [line.strip() for line in txt]
#
#     for image_ind in image_inds:
#         i = 0
#         with open(os.path.join(out_path, str(int(image_ind)) + '.txt'), 'a') as f:
#             for p in prim:
#                 ind = p.split()[0].split('/')[-1].split('.')[0]
#                 if int(ind) == int(image_ind):
#                     i += 1
#                     if int(p.split()[1]) == 2:
#                         conf = p.split()[2]
#                         box2d = p.split()[3:7]
#                         xmin, ymin, xmax, ymax = list(map(str, box2d))
#                         annotation = ' '.join(['Car', conf, xmin, ymin, xmax, ymax]) + '\n'
#                         print(annotation)
#                         f.write(annotation)
#                 else:
#                     prim = prim[i:]
#                     break
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_path", default="/media/dataset/Kitti/object/")
#     parser.add_argument("--train_annotation", default="/home/zhangy/prim_train.txt")
#     parser.add_argument("--test_annotation",  default="/home/zhangy/prim_test.txt")
#     parser.add_argument("--train_file", default="/home/zhangy/yolov3/mAP/predicted_trainprim")
#     parser.add_argument("--test_file", default="/home/zhangy/yolov3/mAP/predicted_prim")
#     flags = parser.parse_args()
#
#     if os.path.exists(flags.train_file): shutil.rmtree(flags.train_file)
#     os.mkdir(flags.train_file)
#     if os.path.exists(flags.test_file): shutil.rmtree(flags.test_file)
#     os.mkdir(flags.test_file)
#
#     convert_voc_annotation(flags.data_path, 'train', flags.train_annotation, flags.train_file)
#     convert_voc_annotation(flags.data_path, 'val', flags.test_annotation, flags.test_file)

###########################
# convert yolov3' pred 2dbox into the form of F-P'prim 2dbox
###########################
# import os
# import argparse
# import shutil
#
#
# def convert_voc_annotation(data_path, data_type, anno_path, out_path):
#
#     classes = ['Car']
#     img_inds_file = os.path.join(data_path, data_type + '.txt')  #
#     # data_path = os.path.join(data_path, 'training')
#     with open(img_inds_file, 'r') as f:
#         txt = f.readlines()
#         image_inds = [line.strip() for line in txt]
#
#     with open(out_path, 'a') as a:
#         for image_ind in image_inds:
#             with open(os.path.join(anno_path, str(int(image_ind)) + '.txt'), 'r') as r:
#                 txt = r.readlines()
#                 pred = [line.strip() for line in txt]
#                 for p in pred:
#                     ind = p.split()[0]
#                     if ind == classes[0]:
#                         conf = p.split()[1]
#                         box2d = p.split()[2:6]
#                         xmin, ymin, xmax, ymax = list(map(str, box2d))
#                         annotation = ' '.join(['dataset/KITTI/object/training/image_2/'+image_ind+'.png', '2',
#                                                conf, xmin, ymin, xmax, ymax]) + '\n'
#                         print(annotation)
#                         a.write(annotation)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_path", default="/media/dataset/Kitti/object/")
#     parser.add_argument("--train_annotation", default="/home/zhangy/pred_train.txt")
#     parser.add_argument("--test_annotation",  default="/home/zhangy/pred_test.txt")
#     parser.add_argument("--train_file", default="/home/zhangy/yolov3/mAP/predicted_train11z_0.1")
#     parser.add_argument("--test_file", default="/home/zhangy/yolov3/mAP/predicted11z_0.1")
#     flags = parser.parse_args()
#
#     if os.path.exists(flags.train_annotation): os.remove(flags.train_annotation)
#     if os.path.exists(flags.test_annotation): os.remove(flags.test_annotation)
#
#     convert_voc_annotation(flags.data_path, 'train', flags.train_file, flags.train_annotation)
#     convert_voc_annotation(flags.data_path, 'val', flags.test_file, flags.test_annotation)

###########################
# convert 2dbox into the form of avod' pred 3dbox
###########################
# import os
# import argparse
#
#
# def convert_voc_annotation(data_path, data_type, anno_path, out_path):
#
#     classes = ['Car']  # , 'Pedestrian', 'Cyclist'
#     img_inds_file = os.path.join(data_path, data_type + '.txt')  #
#     # data_path = os.path.join(data_path, 'training')
#     with open(img_inds_file, 'r') as f:
#         txt = f.readlines()
#         image_inds = [line.strip() for line in txt]
#
#     for image_ind in image_inds:
#         with open(os.path.join(anno_path, str(int(image_ind)) + '.txt'), 'r') as f:
#             txt = f.readlines()
#             prim = [line.strip() for line in txt]
#         with open(os.path.join(out_path, str(image_ind) + '.txt'), 'a') as f:
#             for p in prim:
#                 conf = p.split()[1]
#                 box2d = p.split()[2:6]
#                 xmin, ymin, xmax, ymax = list(map(str, box2d))
#                 annotation = ' '.join(['Car -1 -1 -10.0', xmin, ymin, xmax, ymax, '0 0 0 0 0 0 0',  conf]) + '\n'
#                 print(annotation)
#                 f.write(annotation)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_path", default="/media/dataset/Kitti/object/")
#     parser.add_argument("--annotation", default="/home/zhangy/yolov3/mAP/predicted11z/")
#     parser.add_argument("--file", default="/home/zhangy/avod/avod/data/outputs/pyramid_cars_with_aug_example/predictions/kitti_native_eval/0.1/6733/data/")
#     flags = parser.parse_args()
#
#     convert_voc_annotation(flags.data_path, 'val', flags.annotation, flags.file)
