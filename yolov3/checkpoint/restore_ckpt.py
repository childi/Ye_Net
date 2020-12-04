import os
from tensorflow.python import pywrap_tensorflow

checkpoint_path = os.path.join(r'/media/personal_data/zhangye/previous_results/yolov3_ckpt_xy/yolov3_test_loss=10.6363.ckpt-85')

# checkpoint_path = os.path.join(r'/media/personal_data/zhangye/outputs/pyramid_cars_with_aug_example/checkpoints'
#                                 r'/pyramid_cars_with_aug_example-00006733')
# checkpoint_path1 = os.path.join(r'/media/personal_data/zhangye/outputs/pyramid_cars_with_aug_example/checkpoints'
#                                r'/pyramid_cars_with_aug_example-00047131')

# 从checkpoint中读出数据pyramid_cars_with_aug_example/checkpoints/
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# reader1 = pywrap_tensorflow.NewCheckpointReader(checkpoint_path1)
# reader = tf.train.NewCheckpointReader(checkpoint_path) # 用tf.train中的NewCheckpointReader方法
var_to_shape_map = reader.get_variable_to_shape_map()
# var_to_shape_map1 = reader1.get_variable_to_shape_map()
# key1 = [key1 for key1 in var_to_shape_map1]
i = 0
j = 0
# 输出权重tensor名字和值
for key in var_to_shape_map:
    # print("tensor_name: ", key, reader.get_tensor(key).shape)
    j = j + 1

    # selective output
    if 'conv_mbbox' in key.split('/'):
        i = i + 1
        print("tensor_name: ", key, reader.get_tensor(key).shape)  # , reader.get_tensor(key), reader1.get_tensor(key)
    # if key in key1:
    #     i = i + 1
    #     if (reader.get_tensor(key) == reader1.get_tensor(key)).all():
    #         # pass
    #         i = i + 1
    #         print("same value tensor_name: ", key)  # , reader.get_tensor(key), reader1.get_tensor(key)
    #         # if (reader.get_tensor(key) == 0).all():
    #         #     print('====')
    #     else:
    #         print("not same value tensor_name: ", key, reader.get_tensor(key), reader1.get_tensor(key))  #
    #         # if (reader.get_tensor(key) == 0).all():
    #         #     print('====')
    # else:
    #     print("not in tensor_name: ", key, reader.get_tensor(key).shape)
print(i)
print('all:', j)
