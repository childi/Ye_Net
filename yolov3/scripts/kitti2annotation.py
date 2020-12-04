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
                # or float(label[1]) > 0.5 or int(label[2]) > 2 or float(label[7])-float(label[5]) < 25
                if label[0] not in classes:
                    continue
                bbox = label[4:8]
                class_ind = classes.index(label[0].strip())
                xmin = str(round(float(bbox[0])))
                xmax = str(round(float(bbox[2])))
                ymin = str(round(float(bbox[1])))
                ymax = str(round(float(bbox[3])))
                # alpha = label[3]
                # z = label[13]
                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_ind)])  # , z, alpha
            print(annotation)
            f.write(annotation + "\n")
    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/media/dataset/Kitti/object/")
    parser.add_argument("--train_annotation", default="/home/zhangy/yolov3/data/dataset/kitti_train00.txt")
    parser.add_argument("--test_annotation",  default="/home/zhangy/yolov3/data/dataset/kitti_test00.txt")
    flags = parser.parse_args()

    if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)

    # TODO:split train/val
    # with open('trainval.txt', 'r') as f:
    #     txt = f.readlines()
    #     train_num = round(0.9 * len(txt))
    # train_val.split(train_num)

    num1 = convert_voc_annotation(flags.data_path, 'train', flags.train_annotation)
    num2 = convert_voc_annotation(flags.data_path, 'val', flags.test_annotation)
    print('=> The number of image for train is: %d\tThe number of image for test is:%d' % (num1, num2))
