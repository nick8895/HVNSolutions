import sys
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
import copy
import cv2
import math
from scipy.spatial.transform import Rotation as R
from bullet_env.util import setup_bullet_client, stdout_redirected
from transform.affine import Affine
from image_util import draw_pose
from data_util import store_data_grasp
import random


counter = 0 # Global counter for storing data

def plan_1(bullet_client, base_position, rotation, cube_urdf, quader_urdf):
    """
    Spawnt zwei Würfel nebeneinander mit 0,1 cm Offset und auf diesen Würfeln jeweils einen weiteren Würfel.
    Zusätzlich wird ein Quader (10x5x5) auf die oberen Würfel platziert.
    """
    cube_size = 5  # in cm
    quader_size = (10, 5, 5)  # in cm
    offset = 0.15  # in cm

    rad = math.radians(rotation)
    cos_theta = math.cos(rad)
    sin_theta = math.sin(rad)

    def rotate(x, y):
        return (
            x * cos_theta - y * sin_theta,
            x * sin_theta + y * cos_theta
        )

    def quaternion_from_rotation_z(angle_rad):
        half_angle = angle_rad / 2.0
        return [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]

    positions = [
        (0, 0, 0),                           # Würfel A (unten links)
        (cube_size + offset, 0, 0),          # Würfel B (unten rechts)
        (0, 0, cube_size + offset),          # Würfel C (oben auf A)
        (cube_size + offset, 0, cube_size + 0.1),  # Würfel D (oben auf B)
    ]

    object_ids = []
    for pos in positions:
        rel_x, rel_y, rel_z = pos
        rx, ry = rotate(rel_x, rel_y)
        spawn_position = [
            base_position[0] + rx / 100.0,  
            base_position[1] + ry / 100.0,  
            base_position[2] + rel_z / 100.0
        ]
        spawn_orientation = quaternion_from_rotation_z(rad)
        obj_id = bullet_client.loadURDF(cube_urdf, spawn_position, spawn_orientation)
        object_ids.append([obj_id, spawn_position, spawn_orientation])

    # Quader oben drauf
    quader_x = (cube_size + offset) / 2.0
    quader_z = 2 * (cube_size + offset)
    rx_quader, ry_quader = rotate(quader_x, 0)
    quader_position = [
        base_position[0] + rx_quader / 100.0,
        base_position[1] + ry_quader / 100.0,
        base_position[2] + quader_z / 100.0
    ]
    quader_orientation = quaternion_from_rotation_z(rad)
    quader_id = bullet_client.loadURDF(quader_urdf, quader_position, quader_orientation)
    # Objekt-Infos für den Quader:
    object_ids.append([quader_id, quader_position, quader_orientation])

    return object_ids

def plan_2(bullet_client, base_position, rotation, cube_urdf, quader_urdf):
    """
    Ähnlich wie plan_1, aber baut einen "Torbogen" aus 4 Würfeln und 1 Quader:
    - 2 Türme je 2 Würfel
    - Quader oben als Brücke
    Rückgabe: Liste [ [obj_id, pos, ori], ... ]
    """
    cube_size = 5.0  # cm
    offset = 0.1     # cm (kleiner Spalt)
    gap = 4.0        # cm Abstand zwischen den beiden Türmen
    quader_size = (10, 5, 5)  # cm (Länge x Breite x Höhe)

    # Wir rotieren um die Z-Achse um "rotation" Grad
    rad = math.radians(rotation)
    cos_theta = math.cos(rad)
    sin_theta = math.sin(rad)

    def rotate(x, y):
        return (x * cos_theta - y * sin_theta,
                x * sin_theta + y * cos_theta)

    def quaternion_from_rotation_z(angle_rad):
        half_angle = angle_rad / 2.0
        return [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]

    # --- Positionen definieren ---
    # Linker Turm: unten / oben
    left_bottom  = (0,           0,            0)
    left_top     = (0,           0,            cube_size + offset)  
    # Rechter Turm: unten / oben
    right_bottom = (cube_size + gap, 0,         0)
    right_top    = (cube_size + gap, 0,         cube_size + offset)
    # Quader oben in der Mitte (10 cm lang, soll mit 2,5 cm auf jedem Turm aufliegen)
    # => Die x-Position der Mitte vom Quader = (0 + (cube_size+gap)) / 2
    #                                          = (5 + gap=5) / 2 = 5 cm
    # => Quader-Höhe = 5 cm; Wir legen ihn auf die Oberkante der Würfel (z = 5 + offset).
    #    Unterkante Quader = 5 + offset, sein center => + quader_size[2]/2
    #    => centerZ = 5 + offset + (5/2) = 5 + 0.1 + 2.5 = 7.6 cm
    quader_center_x = (0 + (cube_size + gap)) / 2.0  
    quader_center_z = (cube_size + offset) + (quader_size[2] / 2.0)
    top_quader = (quader_center_x, 0, quader_center_z)

    positions = [left_bottom, left_top, right_bottom, right_top, top_quader]

    # --- Würfel oder Quader? ---
    # Index 0..3 = Würfel, Index 4 = Quader
    object_ids = []
    for i, pos in enumerate(positions):
        rel_x, rel_y, rel_z = pos
        rx, ry = rotate(rel_x, rel_y)
        spawn_position = [
            base_position[0] + rx/100.0,  
            base_position[1] + ry/100.0,  
            base_position[2] + rel_z/100.0
        ]
        spawn_orientation = quaternion_from_rotation_z(rad)

        if i < 4:
            # Würfel
            obj_id = bullet_client.loadURDF(cube_urdf, spawn_position, spawn_orientation)
        else:
            # Quader
            obj_id = bullet_client.loadURDF(quader_urdf, spawn_position, spawn_orientation)

        object_ids.append([obj_id, spawn_position, spawn_orientation])

    return object_ids

def plan_3(bullet_client, base_position, rotation, cube_urdf, quader_urdf):
    """
    Ebene 1:
      - 2 Quader (10 x 5 x 5 cm) nebeneinander, 4 cm Abstand in X-Richtung
    Ebene 2:
      - 1 Quader quer (90° zur ersten Ebene), mittig auf Ebene 1
    Ebene 3:
      - 2 Würfel (5x5x5 cm) übereinander, zentriert auf dem oberen Quader

    Rückgabe: Liste mit 5 Einträgen:
      0: Quader unten links
      1: Quader unten rechts
      2: Quader oben quer
      3: Erster Würfel oben
      4: Zweiter Würfel oben drauf
    Jeder Eintrag: [obj_id, spawn_pos, spawn_orientation]
    """

    # Annahmen zu den Dimensionen
    quader_len  = 10.0  # cm (Länge)
    quader_b    = 5.0   # cm (Breite)
    quader_h    = 5.0   # cm (Höhe)

    cube_size   = 5.0   # cm (Würfelkantenlänge)
    gap         = 2.0   # cm zwischen Quadern der gleichen Ebene
    offset_z    = 0.1   # cm kleiner Spalt zwischen Ebenen
    
    # Globale Rotation um Z
    rad = math.radians(rotation)
    cos_theta = math.cos(rad)
    sin_theta = math.sin(rad)

    def rotate_xy(x, y):
        """Rotiere Punkt (x,y) um 'rotation' Grad in der XY-Ebene."""
        rx = x * cos_theta - y * sin_theta
        ry = x * sin_theta + y * cos_theta
        return rx, ry

    def quaternion_from_rotation_z(angle_rad):
        """Erzeuge ein Quaternion für Rotation um Z."""
        half = angle_rad / 2.0
        return [0.0, 0.0, math.sin(half), math.cos(half)]

    # -----------------------
    # Ebene 1: Zwei Quader parallel (L = 10 cm entlang X)
    #   Quader A bei (0, 0, 0)
    #   Quader B bei (L + gap, 0, 0) => (10 + 4=14, 0, 0)
    #   Höhe = 5 cm
    A_pos = (0.0, 0.0, 0.0)
    B_pos = (quader_len + gap, 0.0, 0.0)

    # -----------------------
    # Ebene 2: Ein Quader quer (90° zur Ebene 1).
    #   Liegt oben auf Ebene 1 => z = 5 + 0.1 = 5.1 cm
    #   Wir wollen ihn mittig zwischen A & B, also x=7 cm
    #   Weil er 10 cm lang ist, und jetzt L=10 cm in Y-Richtung (durch 90°-Drehung).
    #   => Also C=(7, 0, 5.1), orientation = rotation + 90°.
    top_z_2 = quader_h + offset_z  # 5 + 0.1 = 5.1
    C_pos = ((A_pos[0] + B_pos[0]) / 2.0, 0.0, top_z_2)  # x=7, y=0
    
    # -----------------------
    # Ebene 3: Zwei WÜRFEL übereinander (je 5 cm hoch)
    #   Stehen in der Mitte auf dem Quader C.
    #   Also x=7, y=0, z=(Ebene2-Höhe + QuaderHöhe=5) => 10.1? 
    #   Da Quader C hat oben z= 5.1 + 5 cm Höhe = 10.1
    #   Also der erste Würfel: z= 10.1
    #   Der zweite Würfel: z= 15.2 (nochmal +5.1)
    #   Falls URDF den Ursprung UNTER dem Objekt hat, reicht z=10.1 bzw. 15.2
    #   Falls URDF den Ursprung in der Mitte hat, müsstest du +2.5 draufrechnen.
    
    top_cube_1_z = top_z_2 + quader_h  # = 5.1 + 5 = 10.1
    top_cube_2_z = top_cube_1_z + cube_size + offset_z  # 10.1 + 5 + 0.1 = 15.2

    D_pos = (C_pos[0], C_pos[1], top_cube_1_z)
    E_pos = (C_pos[0], C_pos[1], top_cube_2_z)

    # -----------------------
    # Sammele alle Positionen in Reihenfolge:
    #  0: Quader A
    #  1: Quader B
    #  2: Quader C (quer)
    #  3: Würfel D
    #  4: Würfel E
    positions = [A_pos, B_pos, C_pos, D_pos, E_pos]

    object_ids = []
    for i, (px, py, pz) in enumerate(positions):
        # Rotation um Z
        # - Ebene1 Quader: i < 2 => rotation + 0
        # - Ebene2 Quader: i == 2 => rotation + 90
        # - Ebene3 Würfel: i >= 3 => rotation + 0 (oder +0.0)
        extra_angle = 0.0
        if i < 2:  # Quader C
            extra_angle = 90.0  # quer
        
        angle_rad = math.radians(rotation + extra_angle)
        spawn_orientation = quaternion_from_rotation_z(angle_rad)

        # XY verschieben
        rx, ry = rotate_xy(px, py)
        spawn_position = [
            base_position[0] + rx / 100.0,
            base_position[1] + ry / 100.0,
            base_position[2] + pz / 100.0
        ]

        # Laden
        if i < 3:
            # i=0..2 => Quader (A,B,C)
            obj_id = bullet_client.loadURDF(quader_urdf, spawn_position, spawn_orientation)
        else:
            # i=3..4 => Würfel (D,E)
            obj_id = bullet_client.loadURDF(cube_urdf, spawn_position, spawn_orientation)

        object_ids.append([obj_id, spawn_position, spawn_orientation])

    return object_ids

def minimal_angle(angle_deg: float) -> float:
    """
    Mappt jeden Winkel (Grad) in das Intervall [-180, +180].
    """
    return (angle_deg + 180) % 360 - 180

def get_yaw_from_quat(qx, qy, qz, qw) -> float:
    r = R.from_quat([qx, qy, qz, qw])
    # Verwende 'zxy', damit die erste Euler-Komponente genau die Z-Drehung (yaw) ist.
    yaw, _, _ = r.as_euler('zxy', degrees=True)
    return minimal_angle(yaw)


def pick_and_place_object(bullet_client, robot, obj_pos, obj_ori, place_position, _rotation, i, task_info, observations, pose,
                           dataset_directory, store_dataset, camera_factory):
    """
    Greift das Objekt so, dass seine Yaw orientiert wird wie obj_ori (minimaler Drehweg),
    und legt es dann bei place_position ab.
    """
    # 1) Yaw aus Objekt-Orientierung
    obj_yaw = get_yaw_from_quat(*obj_ori)
    print("obj_yaw:", obj_yaw)

    # 2) Hier KEIN zusätzlicher Versatz mehr. Wir wollen 'obj_yaw' = final Yaw
    target_yaw = minimal_angle(obj_yaw)

    print("target_yaw", target_yaw)

    # 3) Falls dein Greifer in der URDF bereits "richtig" nach unten zeigt,
    #    kannst du base_rot = identity lassen. Sonst,
    #    falls der Greifer standardmäßig +90° versetzt ist, könntest du
    #    z.B. base_rot = R.from_euler('XYZ', [0, 0, -90], degrees=True).
    base_rot = R.from_euler('XYZ', [180, 0, 90], degrees=True)
    
    # 4) End-Effektordrehung um Z
    z_rot = R.from_euler('Z', target_yaw, degrees=True)
    # final_rot = base_rot * z_rot   # ALT, oft um 90° verschoben
    final_rot = z_rot * base_rot     # NEU
    final_quat = final_rot.as_quat()


    # 5) Pose zusammenbauen
    z_affine = Affine(rotation=final_quat)
    obj_pose_affine = Affine(translation=obj_pos) * z_affine

    # Approach-Offsets (greift von oben, negative z geht "abwärts")
    pre_grasp_offset = Affine([0, 0, -0.20])
    grasp_offset     = Affine([0, 0, -0.01]) 
    place_offset     = Affine([0, 0, -0.02])

    pre_grasp_pose = obj_pose_affine * pre_grasp_offset
    grasp_pose     = obj_pose_affine * grasp_offset

    # Ablage-Pose
    place_pose_affine = Affine(translation=place_position) * z_affine
    pre_place_pose    = place_pose_affine * pre_grasp_offset
    place_pose        = place_pose_affine * place_offset

    # Robot move
        
    global counter

    robot.ptp(pre_grasp_pose)

    observations = [cam.get_observation() for cam in camera_factory]
    if store_dataset:
        store_data_grasp(counter, task_info, observations, pose, dataset_directory)
        counter+= 1
    robot.lin(grasp_pose)
    robot.gripper.close()

    observations = [cam.get_observation() for cam in camera_factory]
    if store_dataset:
        store_data_grasp(counter, task_info, observations, pose, dataset_directory)
        counter+= 1
    robot.lin(pre_grasp_pose)

    observations = [cam.get_observation() for cam in camera_factory]
    if store_dataset:
        store_data_grasp(counter, task_info, observations, pose, dataset_directory)
        counter+= 1
    robot.ptp(pre_place_pose)

    observations = [cam.get_observation() for cam in camera_factory]
    if store_dataset:
        store_data_grasp(counter, task_info, observations, pose, dataset_directory)
        counter+= 1
    robot.lin(place_pose)
    robot.gripper.open()

    observations = [cam.get_observation() for cam in camera_factory]
    if store_dataset:
        store_data_grasp(counter, task_info, observations, pose, dataset_directory)
        counter+= 1
    robot.lin(pre_place_pose)
    observations = [cam.get_observation() for cam in camera_factory]
    if store_dataset:
        store_data_grasp(counter, task_info, observations, pose, dataset_directory)
        counter+= 1


def quaternion_from_rotation_z(angle_rad):
    half_angle = angle_rad / 2.0
    return [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]


def cleanup_grasp_task(bullet_client, robot, task, base_drop_position, rotation,i, task_info, observations, pose, dataset_directory, store_dataset, camera_factory):
    """
    Greift alle Objekte in umgekehrter Reihenfolge und legt sie an base_drop_position ab,
    parallel zur Y-Achse versetzt.
    """
    y_offset = 0.125

    for i, grasp_obj in enumerate(reversed(task.grasp_objects)):
        drop_position = [
            base_drop_position[0],
            base_drop_position[1] + i * y_offset,
            base_drop_position[2]
        ]

        # Pose aus dem Task-Objekt
        obj_pos = grasp_obj.pose[:3, 3]
        rotation_matrix = grasp_obj.pose[:3, :3]
        obj_ori = R.from_matrix(rotation_matrix).as_quat()

        print("Objekt-Orientierung:", obj_ori)

        # Ausführung
        pick_and_place_object(
            bullet_client, 
            robot, 
            obj_pos,
            obj_ori,
            drop_position,
            rotation,
            i,
            task_info,
            observations,
            pose,
            dataset_directory,
            store_dataset,
            camera_factory
        )

    robot.gripper.open()

def execute_random_plan(bullet_client, spawn_position, rotation, cube_urdf, quader_urdf_path):
    choice = random.choice([1, 2, 3])
    if choice == 1:
        print("Running plan_1")
        object_ids = plan_1(
            bullet_client,
            spawn_position,
            rotation,
            cube_urdf,
            quader_urdf_path
        )
    elif choice == 2:
        print("Running plan_2")
        object_ids = plan_2(
            bullet_client,
            spawn_position,
            rotation,
            cube_urdf,
            quader_urdf_path
        )
    else:
        print("Running plan_3")
        object_ids = plan_3(
            bullet_client,
            spawn_position,
            rotation,
            cube_urdf,
            quader_urdf_path
        )
    return object_ids

@hydra.main(version_base=None, config_path="config", config_name="tn_train_data")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    bullet_client = setup_bullet_client(cfg.render)

    env = instantiate(cfg.env, bullet_client=bullet_client)
    robot = instantiate(cfg.robot, bullet_client=bullet_client)

    t_bounds = copy.deepcopy(robot.workspace_bounds)
    t_bounds[2, 1] = t_bounds[2, 0]

    task_factory = instantiate(cfg.task_factory, t_bounds=t_bounds)
    oracle = instantiate(cfg.oracle)

    t_center = np.mean(t_bounds, axis=1)
    camera_factory = instantiate(cfg.camera_factory, bullet_client=bullet_client, t_center=t_center)

    # Assets
    cube_urdf = "/home/jovyan/data/assets/objects/cube/object.urdf"
    quader_urdf_path = "/home/jovyan/data/assets/objects/quader/object.urdf"


    


    function_params = []

    for i in range(cfg.n_scenes):

            # Random spawn position
        random_x = np.random.choice(np.arange(0.6, 0.8, 0.05))
        random_y = np.random.choice(np.arange(-0.25, 0.25, 0.05))
        spawn_position = [random_x, random_y, 0.0]
        
        # Random rotation
        rotation = np.random.choice(np.arange(0, 359, 30))

        object_ids = execute_random_plan(bullet_client, spawn_position, rotation, cube_urdf, quader_urdf_path)  

        robot.home()
        robot.gripper.open()

        # Task zusammenbauen
        object_specs = []
        for idx, (pyb_id, pos, ori) in enumerate(object_ids):
            my_type = "cube" if idx < 4 else "quader"
            spec = {
                "object_type": my_type,
                "object_id": pyb_id,
                "position": pos,
                "orientation": ori
            }
            object_specs.append(spec)

        task = task_factory.create_task_from_specs(object_specs)

        # Alle Objekte in einer Reihe ablegen
        drop_position =  [0.35, -0.28, 0.0]
        

        # (Optional) Infos/Oracle
        task_info = task.get_info()
        pose = oracle.solve(task)

        # Kamerabilder
        observations = [cam.get_observation() for cam in camera_factory.cameras]
        cleanup_grasp_task(bullet_client, robot, task, drop_position, rotation,i,
                            task_info, observations, pose, cfg.dataset_directory, cfg.store_dataset, camera_factory.cameras)
    

        # Debug/Visualisierung
        if cfg.debug:
            image_copy = copy.deepcopy(observations[0]['rgb'])
            depth_copy = copy.deepcopy(observations[0]['depth'])
            draw_pose(observations[0]['extrinsics'], pose, observations[0]['intrinsics'], image_copy)
            #cv2.imshow('rgb', image_copy)
            depth_copy = depth_copy / 2.0
            #cv2.imshow('depth', depth_copy)

            #key_pressed = cv2.waitKey(0)
            #if key_pressed == ord('q'):
            #    break

        # Task entfernen
        task.clean(env)

    cv2.destroyAllWindows()
    with stdout_redirected():
        bullet_client.disconnect()


if __name__ == "__main__":
    main()
