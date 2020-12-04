import matplotlib.pyplot as plt

# with open('/home/zhangy/map.txt', 'r') as file:
#   ann = file.readlines()
#   steps = []
#   mAP = []
#   recalls = []
#   for i in range(len(ann)):
#     if i % 5 == 0:
#       step = ann[i].split()[1].split(',')[0]
#       steps.append(int(step))
#     elif i % 5 == 1:
#       map = ann[i].split('   ')[1].split()[0].split('%')[0]
#       mAP.append(float(map))
#     elif i % 5 == 2:
#       recall = ann[i].split()[2].split('%')[0]
#       recalls.append(float(recall))
#
# with open('/home/zhangy/ap.txt', 'r') as file:
#   anna = file.readlines()
#   AP2 = []
#   APb = []
#   AP3 = []
#   PED = []
#   CYC = []
#   for i in range(len(anna)):
#     if anna[i].split()[0] == 'car_detection':
#       ap2 = anna[i].split()[3]
#       AP2.append(float(ap2))
#     if anna[i].split()[0] == 'car_detection_BEV':
#       apb = anna[i].split()[3]
#       APb.append(float(apb))
#     if anna[i].split()[0] == 'car_detection_3D':
#       ap3 = anna[i].split()[3]
#       AP3.append(float(ap3))
#     if anna[i].split()[0] == 'pedestrian_detection_3D':
#       ped = anna[i].split()[3]
#       PED.append(float(ped))
#     if anna[i].split()[0] == 'cyclist_detection_3D':
#       cyc = anna[i].split()[3]
#       CYC.append(float(cyc))
#
# plt.plot(steps, mAP, color='blue', linewidth=1.0, linestyle='-')
# plt.plot(steps, recalls, color='green', linewidth=1.0, linestyle='-')
# plt.plot(steps, AP2, color='orange', linewidth=1.0, linestyle='--')
# plt.plot(steps, APb, color='purple', linewidth=1.0, linestyle='--')
# plt.plot(steps, AP3, color='red', linewidth=1.0, linestyle='--')
# plt.plot(steps, PED, color='black', linewidth=1.0, linestyle=':')
# plt.plot(steps, CYC, color='cyan', linewidth=1.0, linestyle=':')
# plt.show()

steps = [i*6733 for i in range(1, 51)]
file_name = ['ap11.txt', 'apzz.txt', 'ap11_mutiply.txt', 'apzz2dupdate.txt', 'ap32zz.txt', 'apgtz.txt', 'ap_az1.txt']
color = ['black', 'red', 'orange', 'green', 'blue', 'purple', 'yellow']
line_num = 4
AP2 = [[] for i in range(line_num)]
APb = [[] for i in range(line_num)]
AP3 = [[] for i in range(line_num)]
for n in range(line_num):
  with open('/home/zhangy/' + file_name[n], 'r') as file:
    anna = file.readlines()
    for i in range(len(anna)):
      if anna[i].split()[0] == 'car_detection':
        ap2 = anna[i].split()[3]
        AP2[n].append(float(ap2))
      if anna[i].split()[0] == 'car_detection_BEV':
        apb = anna[i].split()[3]
        APb[n].append(float(apb))
      if anna[i].split()[0] == 'car_detection_3D':
        ap3 = anna[i].split()[3]
        AP3[n].append(float(ap3))
  plt.plot(steps, AP2[n][:], color=color[n], linewidth=1.0, linestyle=':')
  plt.plot(steps, APb[n][:], color=color[n], linewidth=1.0, linestyle='--')
  plt.plot(steps, AP3[n][:], color=color[n], linewidth=1.0, linestyle='-')
plt.show()
