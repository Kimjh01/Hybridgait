# import glob, pickle, numpy as np
# pose_list = glob.glob(r"D:\Grew_reduce_pose_pkl\**\*.pkl", recursive=True)
# print("PKL 개수:", len(pose_list))
# for p in pose_list[:100]:       # 세 개만 확인
#     print(p, np.array(pickle.load(open(p,'rb'))).shape)

import torch, sys

print(torch.__version__)
print(torch.cuda.is_available())

