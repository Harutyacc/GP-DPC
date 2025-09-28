import numpy as np
import scipy.io as sio
from data_generator import data_generator
import matplotlib.pyplot as plt

def mat_to_numpy(mat_file):
  
    data = sio.loadmat(mat_file)
    
    structs_data = {}
    
    # 遍历文件中的所有变量
    for key in data:
      
      # 检查变量是否是结构体
      if isinstance(data[key], np.ndarray) and data[key].ndim == 2:
        
        # 提取字段
        if 's' in data[key].dtype.names:
          structs_data[key] = {
            's' : data[key]['s'][0][0],
            'sigma' : data[key]['sigma'][0][0],
            'k' : data[key]['k'][0][0],
          }
    
    return structs_data
