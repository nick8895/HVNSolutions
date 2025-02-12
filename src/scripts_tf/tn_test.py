import sys
import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from hydra.utils import instantiate
import copy
import math
from scipy.spatial.transform import Rotation as R
import os
import cv2
from bullet_env.util import setup_bullet_client
from transform.affine import Affine
from convert_dataset import create_pointcloud, transform_pointcloud, filter_workspace, create_heightmap
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.transporter_network.model import TransporterNetworkAttention
#from src.scripts_bullet.data_util import store_data_grasp #TODO: make sure, import works




def minimal_angle(a): return (a + 180) % 360 - 180


def get_yaw_from_quat(qx,qy,qz,qw): return minimal_angle(R.from_quat([qx,qy,qz,qw]).as_euler('zxy',True)[0])
def quaternion_from_rotation_z(a): return [0,0,math.sin(a/2),math.cos(a/2)]



def inference_and_pick(bc, robot, model, rgb):  #TODO: Fix function to use right inputs
    x_tf = tf.cast(tf.expand_dims(rgb, 0), tf.float32)
    pred = model(x_tf)[0].numpy()
    dx, dy, dz, dtheta = pred
    px, py, pz = 0.6 + dx, 0 + dy, 0.02 + dz
    ori = quaternion_from_rotation_z(math.radians(dtheta))
    pick = Affine([px, py, pz], ori)
    robot.ptp(pick * Affine([0, 0, 0.15]))
    robot.lin(pick)
    robot.gripper.close()
    robot.lin(pick * Affine([0, 0, 0.15]))
    place = Affine([0.4, -0.2, 0.02])
    robot.ptp(place)
    robot.gripper.open()
    robot.ptp(pick * Affine([0, 0, 0.15]))

@hydra.main(version_base=None, config_path="config", config_name="tn_train_data")
def test_transporter_network(cfg: DictConfig) -> None:
    bc = setup_bullet_client(cfg.render)
    env = instantiate(cfg.env, bullet_client=bc)
    robot = instantiate(cfg.robot, bullet_client=bc)
    t_bounds = copy.deepcopy(robot.workspace_bounds)
    t_bounds[2, 1] = t_bounds[2, 0]
    task_factory = instantiate(cfg.task_factory, t_bounds=t_bounds)
    oracle = instantiate(cfg.oracle)
    cf = instantiate(cfg.camera_factory, bullet_client=bc, t_center=np.mean(t_bounds, axis=1))
    test_path = "/home/jovyan/data/datasets/test_data"
    cube_urdf = "/home/jovyan/data/assets/objects/cube/object.urdf"
    spawn_pos = [0.7, 0, 0]
    bc.loadURDF(cube_urdf, spawn_pos, quaternion_from_rotation_z(0))
    robot.home()
    robot.gripper.open()

    model = TransporterNetworkAttention(64, 18)

    checkpoint_prefix = "/home/jovyan/data/models/tn_model/model"
    index_file = checkpoint_prefix + ".index"
    data_file = checkpoint_prefix + ".data-00000-of-00001"

    if not (os.path.exists(index_file) and os.path.exists(data_file)):
        raise FileNotFoundError(f"Checkpoint files not found: {index_file} or {data_file}. Ensure both files are present.")

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(checkpoint_prefix).expect_partial()

    task = task_factory.create_task_from_specs([
        {"object_type": "cube", "object_id": 0, "position": spawn_pos, "orientation": quaternion_from_rotation_z(0)}
    ])
    oracle.solve(task)

    

    #TODO: make sure, all 3 cams are used, otherweise shadows appear
    obs= [cam.get_observation() for cam in cf.cameras]
   
    intrinsics = obs[0]['intrinsics']
    extrinsics = obs[0]['extrinsics']

    depth_copy = copy.deepcopy(obs[0]['depth'])
    image_copy = copy.deepcopy(obs[0]['rgb'])

    cv2.imshow('Image', image_copy)
    cv2.imshow('Depth', depth_copy)
    cv2.waitKey(0)

    #checken warum nur ein kamerabild verwendet wird, schatten auf RGB zu sehen 
    points, colors = create_pointcloud(image_copy, depth_copy, intrinsics)
    points = transform_pointcloud(points, extrinsics)
    points, colors = filter_workspace(points, colors, cfg.workspace_bounds)
    heightmap, colormap = create_heightmap(points, colors, [256,256], cfg.workspace_bounds)

    # Show processed images
    cv2.imshow('Heightmap', (heightmap - cfg.workspace_bounds[2][0]) / (cfg.workspace_bounds[2][1] - cfg.workspace_bounds[2][0]))
    cv2.imshow('Colormap', colormap)
    cv2.waitKey(0)

    #store_data_grasp(scene_id, task_info, observations, grasp_pose, dataset_directory=test_path) #TODO: give right inputs
   
    predict = model.get_max_locations(heightmap)
    print(model.get_max_locations(heightmap))
    robot.ptp((Affine(predict)))



    # Use processed images for inference
    inference_and_pick(bc, robot, model, heightmap, colormap)  #TODO: Fix this line 

    
    model.get_max_locations(heightmap)
    print(model.get_max_locations(heightmap))

    task.clean(env)
    bc.disconnect()

if __name__ == "__main__":
    test_transporter_network()