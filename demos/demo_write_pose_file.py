"""Calculate camera pose and write to a file inside the original kitti folder"""
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import pykitti
import os

def write_file(filename, path, Ts):
    out_path = os.path.join(path, filename)
    with open(out_path, 'w') as f:
        f.write( "\n".join( " ".join(map(str, x.reshape(-1)[:12])) for x in Ts) )

def generate_pose_file(basedir, date, drive, seq_path):
    # Load the data. Optionally, specify the frame range to load.
    dataset = pykitti.raw(basedir, date, drive)
    # dataset = pykitti.raw(basedir, date, drive, frames=range(0, 5, 5))

    ## loop over all frames, calculate camera pose and write to a file
    T_cam2_imu = dataset.calib.T_cam2_imu
    T_imu_cam2 = np.linalg.inv(T_cam2_imu)
    T_cam3_imu = dataset.calib.T_cam3_imu
    T_imu_cam3 = np.linalg.inv(T_cam3_imu)
    T_velo_imu = dataset.calib.T_velo_imu
    T_imu_velo = np.linalg.inv(T_velo_imu)

    T_w_imus = []

    T_w_cam2s = []
    T_cam20_cam2s = []

    T_w_cam3s = []
    T_cam30_cam3s = []

    T_w_velos = []
    T_velo0_velos = []
    for oxt in dataset.oxts:
        T_w_imus.append(oxt.T_w_imu)

        T_w_cam2 = np.dot(oxt.T_w_imu, T_imu_cam2)
        T_w_cam2s.append(T_w_cam2)
        T_cam20_cam2s.append( np.dot(np.linalg.inv(T_w_cam2s[0]), T_w_cam2) )

        T_w_cam3 = np.dot(oxt.T_w_imu, T_imu_cam3)
        T_w_cam3s.append(T_w_cam3)
        T_cam30_cam3s.append( np.dot(np.linalg.inv(T_w_cam3s[0]), T_w_cam3) )

        T_w_velo = np.dot(oxt.T_w_imu, T_imu_velo)
        T_w_velos.append(T_w_velo)
        T_velo0_velos.append( np.dot(np.linalg.inv(T_w_velos[0]), T_w_velo) )


    out_file_folder = os.path.join(seq_path, 'poses')
    if not os.path.exists(out_file_folder):
        os.mkdir(out_file_folder)

    out_file = 'imu.txt'
    write_file(out_file, out_file_folder, T_w_imus)

    out_file = 'cam_02.txt'
    write_file(out_file, out_file_folder, T_w_cam2s)

    out_file = 'cam_03.txt'
    write_file(out_file, out_file_folder, T_w_cam3s)
    
    out_file = 'velo.txt'
    write_file(out_file, out_file_folder, T_w_velos)


# Change this to the directory where you store KITTI data
basedir = '/media/sda1/minghanz/datasets/kitti/kitti_data'

# # Specify the dataset to load
# date = '2011_09_26'
# drive = '0001'

## loop over all sequences
dates = os.listdir(basedir)
dates = [date for date in dates if os.path.isdir(os.path.join(basedir, date) ) ]

for date in dates:
    date_path = os.path.join(basedir, date)
    seqs = os.listdir(date_path)
    seqs = [seq for seq in seqs if os.path.isdir(os.path.join(date_path, seq))]
    for seq in seqs:
        seq_path = os.path.join(date_path, seq)
        seq_n = seq.split('_drive_')[1].split('_')[0]
        
        print(date, seq_n)
        generate_pose_file(basedir, date, seq_n, seq_path)