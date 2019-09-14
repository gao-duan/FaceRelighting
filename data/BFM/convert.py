#!/usr/bin/env python
# encoding: utf-8

import scipy.io as sio
import numpy as np
import struct 

print("Loading mat files.")
model_info = sio.loadmat('./3DDFA/model_info.mat',matlab_compatible=True)
bfm = sio.loadmat('./BaselFaceModel/01_MorphableModel.mat',matlab_compatible=True)
model_expr = sio.loadmat('./3DDFA/Model_Expression.mat',matlab_compatible=True)
bfm_attrib = sio.loadmat('./BaselFaceModel/04_attributes.mat', matlab_compatible=True)

print("Processing.")
trimIndex = model_info["trimIndex"].astype(np.int32)
trimIndex_f = np.concatenate([3 * trimIndex-3, 3 * trimIndex -2, 3 * trimIndex - 1], axis = 1)
trimIndex_f = trimIndex_f.reshape(-1, )

Model = {}

Model["shapeEV"] = bfm["shapeEV"]              
Model["shapeMU"] = bfm["shapeMU"][trimIndex_f]
Model["shapePC"] = bfm["shapePC"][trimIndex_f] 

Model["texEV"] = bfm["texEV"]                 
Model["texMU"] = bfm["texMU"][trimIndex_f]     
Model["texPC"] = bfm["texPC"][trimIndex_f]    

Model["tl"] = model_info["tri"].T[:,::-1] - 1  
Model["segbin"] = model_info["segbin"][trimIndex.reshape(-1,) - 1] 

Model["expMU"] = model_expr["mu_exp"] 
Model["expPC"] = model_expr["w_exp"]
Model["expEV"] = model_expr["sigma_exp"]

Model["attrib"] = bfm_attrib["age_shape"]
Model["attrib"] = np.append(Model["attrib"], bfm_attrib["age_tex"])
Model["attrib"] = np.append(Model["attrib"], bfm_attrib["gender_shape"])
Model["attrib"] = np.append(Model["attrib"], bfm_attrib["gender_tex"])
Model["attrib"] = np.append(Model["attrib"], bfm_attrib["height_shape"])
Model["attrib"] = np.append(Model["attrib"], bfm_attrib["height_tex"])
Model["attrib"] = np.append(Model["attrib"], bfm_attrib["weight_shape"])
Model["attrib"] = np.append(Model["attrib"], bfm_attrib["weight_tex"])
 
# split a big array into several chunks
# avoiding memory error
def chunks(lst, n):
    n = max(n, 1)
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

chunk_size = 10000

# save array to file in binary mode
def write_array(f, arr, format='f'):
    x,y = arr.shape
    arr = arr.flatten().tolist()
    
    f.write(struct.pack('<i',x))
    f.write(struct.pack('<i',y))
    arr_list = chunks(arr, int(len(arr)/chunk_size))
    for chunk_arr in arr_list:
        f.write(struct.pack('<' + str(len(chunk_arr)) + format, *chunk_arr))
 
print("Writing bfm_data.")
f = open('bfm_data','wb')
write_array(f, Model['shapeEV'])
write_array(f, Model['shapeMU'])
write_array(f, Model['shapePC'])
write_array(f, Model['texEV'])
write_array(f, Model['texMU'])
write_array(f, Model['texPC'])
write_array(f, Model['tl'], 'i')     
write_array(f, Model['segbin'], 'i')
f.close()

print("Writing bfm_exp.")
f2 = open('bfm_exp','wb')
write_array(f2, Model['expEV'])
write_array(f2, Model['expMU'])
write_array(f2, Model['expPC'])
f2.close()

print("Writing bfm_attrib.txt.")
with open('bfm_attrib.txt','w+') as f3:
    print(Model['attrib'].shape)
    for item in Model['attrib']:
        f3.write(str(item) + '\n')

print('Done.')
