import sys
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
import copy
import cv2
from scipy.spatial.transform import Rotation as R

from bullet_env.util import setup_bullet_client, stdout_redirected
from transform.affine import Affine

from image_util import draw_pose
from data_util import store_data_grasp
import math

def plan_1(bullet_client, base_position, rotation, cube_urdf, quader_urdf):
    """
    Spawnt zwei Würfel nebeneinander mit 0,5 cm Offset und auf diesen Würfeln jeweils einen weiteren Würfel.
    Zusätzlich wird ein Quader (10x5x5) auf die oberen Würfel platziert.
    :return: Liste mit den IDs der gespawneten Objekte (Reihenfolge: [unten, unten, oben, oben, quader]).
    """
    cube_size = 5  # in cm
    quader_size = (10, 5, 5)  # in cm
    offset = 0.5  # in cm

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
        (0, 0, 0),                 # Würfel A (unten links)
        (cube_size + offset, 0, 0),# Würfel B (unten rechts)
        (0, 0, cube_size + offset),# Würfel C (oben auf A)
        (cube_size + offset, 0, cube_size + offset), # Würfel D (oben auf B)
    ]

    object_ids = []
    for pos in positions:
        rel_x, rel_y, rel_z = pos
        rx, ry = rotate(rel_x, rel_y)
        spawn_position = [
            base_position[0] + rx / 100.0,  # cm -> m
            base_position[1] + ry / 100.0,  # cm -> m
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
    object_ids.append([obj_id, spawn_position, spawn_orientation])

    #print("Objekt 1")
    #print(object_ids[0][0])
    #print(object_ids[0][1])
    #print(object_ids[0][2])

    #print("Objekt 2")
    #print(object_ids[1][0])
    #print(object_ids[1][1])
    #print(object_ids[1][2])

    #print("Objekt 3")
    #print(object_ids[2][0])
    #print(object_ids[2][1])
    #print(object_ids[2][2])

    #print("Objekt 4")
    #print(object_ids[3][0])
    #print(object_ids[3][1])
    #print(object_ids[3][2])

    #print("Objekt 5")
    #print(object_ids[4][0])
    #print(object_ids[4][1])
    #print(object_ids[4][2])
    return object_ids

def pick_and_place_object(bullet_client, robot, obj_pos, obj_ori, place_position, rotation):
    """
    Greift das Objekt mit einer einfachen "Pick von oben"-Pose und legt es an der place_position ab.
    :param bullet_client: PyBullet-Client
    :param robot: Roboterobjekt
    :param obj_pos: Position des Objekts [x, y, z]
    :param obj_ori: (ALT) Orientierung des Objekts, wird aber jetzt ignoriert, weil wir "von oben" greifen
    :param place_position: [x, y, z] in Metern, wo das Objekt abgelegt werden soll
    :param rotation: Rotation um die z-Achse (in Grad)
    """
import math
from scipy.spatial.transform import Rotation as R

def pick_and_place_object(bullet_client, robot, obj_pos, obj_ori, place_position, rotation):
    """
    Greift das Objekt mit einer "Top-down"-Pose, wobei wirklich um die globale Z rotiert wird.
    """
    # 1) Basis-Rotation, damit der Endeffektor tatsächlich nach unten zeigt
    #    (Anpassen, falls Ihr Roboter anders orientiert ist!)
    base_rot = R.from_euler('XYZ', [180, 0, 0], degrees=True)

    # 2) Gewünschte z-Achs-Drehung (in Weltkoordinaten)
    z_rot = R.from_euler('Z', rotation, degrees=True)

    # 3) Kombinieren
    final_rot = base_rot * z_rot
    final_quat = final_rot.as_quat()

    # 4) Affine aufbauen
    z_affine = Affine(rotation=final_quat)
    obj_pose_affine = Affine(translation=obj_pos) * z_affine

    # Pre-/Post-Grasp-Offsets
    pre_grasp_offset = Affine(translation=[0, 0, -0.20])  # 20 cm über dem Objekt
    grasp_offset     = Affine(translation=[0, 0, -0.02])  # 2 cm über dem Greifpunkt
    place_offset     = Affine(translation=[0, 0, -0.02])  # 2 cm über der Ablageposition

    # Greif-Pose
    pre_grasp_pose = obj_pose_affine * pre_grasp_offset
    grasp_pose = obj_pose_affine * grasp_offset

    # Ablage-Pose
    place_pose_affine = Affine(translation=place_position) * z_affine
    pre_place_pose = place_pose_affine * pre_grasp_offset
    place_pose = place_pose_affine * place_offset

    # Bewegungsabfolge
    robot.ptp(pre_grasp_pose)
    robot.lin(grasp_pose)
    robot.gripper.close()
    robot.lin(pre_grasp_pose)

    robot.ptp(pre_place_pose)
    robot.lin(place_pose)
    robot.gripper.open()
    robot.lin(pre_place_pose)


def quaternion_from_rotation_z(angle_rad):
    half_angle = angle_rad / 2.0
    return [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]

def cleanup_grasp_task(bullet_client, robot, task, base_drop_position, rotation):
    """
    Greift alle Objekte in umgekehrter Reihenfolge und legt sie an der definierten Basisposition ab.
    :param bullet_client: PyBullet-Client
    :param robot: Roboterobjekt
    :param task: GraspTask mit den zu bewegenden Objekten
    :param base_drop_position: [x, y, z] der Basisposition für die Ablage
    :param rotation: Rotation um die z-Achse in Grad
    """
    # Offset für die Ablage (jedes Objekt wird leicht versetzt abgelegt)
    x_offset = 0.05

    for i, grasp_obj in enumerate(reversed(task.grasp_objects)):
        # Zielposition für die Ablage (mit Offset)
        drop_position = [
            base_drop_position[0] + i * x_offset,
            base_drop_position[1],
            base_drop_position[2]
        ]

        # Translation und Orientierung aus grasp_obj.pose extrahieren
        obj_pos = grasp_obj.pose[:3, 3]                 # => numpy-Array, z.B. [x, y, z]
        rotation_matrix = grasp_obj.pose[:3, :3]
        obj_ori = R.from_matrix(rotation_matrix).as_quat()  # => [qx, qy, qz, qw]


        print("Objekt-Position:", obj_pos)
        print("Objekt-Orientierung:", obj_ori)
        print("Ablage-Position:", drop_position)


        # Roboteransteuerung
        pick_and_place_object(
            bullet_client, 
            robot, 
            obj_pos,      # Affine Translation des Objekts
            obj_ori,      # Affine Rotation des Objekts
            drop_position,
            rotation
        )

    # Greifer öffnen
    robot.gripper.open()




@hydra.main(version_base=None, config_path="config", config_name="tn_train_data")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    # PyBullet-Client (GUI oder Headless) einrichten
    bullet_client = setup_bullet_client(cfg.render)

    # Environment und Roboter laden
    env = instantiate(cfg.env, bullet_client=bullet_client)
    robot = instantiate(cfg.robot, bullet_client=bullet_client)

    # Workspace-Bounds anpassen (z-Bereich auf Bodenhöhe setzen)
    t_bounds = copy.deepcopy(robot.workspace_bounds)
    t_bounds[2, 1] = t_bounds[2, 0]

    # Task-Factory und Oracle bereitstellen
    task_factory = instantiate(cfg.task_factory, t_bounds=t_bounds)
    oracle = instantiate(cfg.oracle)

    # Kameras instanziieren (Position/Intrinsics gem. Config)
    t_center = np.mean(t_bounds, axis=1)
    camera_factory = instantiate(cfg.camera_factory, bullet_client=bullet_client, t_center=t_center)

    # Feste Objekte manuell in die Szene laden
    cube_urdf = "/home/jovyan/data/assets/objects/cube/object.urdf"
    quader_urdf_path = "/home/jovyan/data/assets/objects/quader/object.urdf"

    # Basisposition und Rotation
    spawn_position = [0.7, 0.0, 0.0]  # [x, y, z] in Metern
    rotation = 90                    # Drehung um Z-Achse in Grad

    # Erzeuge die Szene
    object_ids = plan_1(
        bullet_client,
        spawn_position,
        rotation,
        cube_urdf,
        quader_urdf_path
    )


    for i in range(cfg.n_scenes):
        # Roboter initialisieren
        robot.home()
        robot.gripper.open()

        #Objekte per plan_1 erzeugen
        object_ids = plan_1(
            bullet_client,
            spawn_position,
            rotation,
            cube_urdf,
            quader_urdf_path
        )
        # object_ids sieht so aus:
        # [
        #   [pyb_id_0, pos0, orientation0],
        #   [pyb_id_1, pos1, orientation1],
        #   ...
        # ]

        # Jetzt baust du dir die Specs zusammen:
        # Für Würfel: object_type = "cube", ...
        # Für Quader: object_type = "quader" (je nachdem, was in
        #   deinem Ordner objects_root/<object_type> liegt)
        object_specs = []
        for i, (pyb_id, pos, ori) in enumerate(object_ids):
            if i < 4:
                my_type = "cube"   # Beispiel: 4 Würfel
            else:
                my_type = "quader" # der 5. Eintrag

            spec = {
                "object_type": my_type,
                "object_id": pyb_id,  # PyBullet-ID übernehmen
                "position": pos,      # das aus plan_1
                "orientation": ori    # das aus plan_1
            }
            object_specs.append(spec)

        # Nun rufst du anstelle von create_task() dein create_task_from_specs auf
        task = task_factory.create_task_from_specs(object_specs)

        # Danach: manuelles Absetzen in einer Reihe
        drop_position = [0.4, 0.0, 0.0]
        cleanup_grasp_task(bullet_client, robot, task, drop_position, rotation)

        # Task-spezifische Aufräumaktion (Task selbst entfernen)
        #task.clean(env)

        # Hole ggf. Task-Infos
        task_info = task.get_info()

        # Die Greif-Pose über das Oracle bestimmen (falls benötigt)
        pose = oracle.solve(task)

        # Kamerabilder aufnehmen
        observations = [camera.get_observation() for camera in camera_factory.cameras]

        # (Optional) Daten speichern
        if cfg.store_dataset:
            store_data_grasp(i, task_info, observations, pose, cfg.dataset_directory)

        # Debug-/Visualisierungsmodus
        if cfg.debug:
            image_copy = copy.deepcopy(observations[0]['rgb'])
            depth_copy = copy.deepcopy(observations[0]['depth'])

            draw_pose(observations[0]['extrinsics'], pose, observations[0]['intrinsics'], image_copy)
            cv2.imshow('rgb', image_copy)
            depth_copy = depth_copy / 2.0
            cv2.imshow('depth', depth_copy)

            key_pressed = cv2.waitKey(0)
            if key_pressed == ord('q'):
                break

            # Beispiel: Greifaktion demonstrieren (nur als Demo)
            env.spawn_coordinate_frame(pose)
            action = Affine.from_matrix(pose)
            pre_grasp_offset = Affine([0, 0, -0.1])
            pre_grasp_pose = action * pre_grasp_offset

            robot.ptp(pre_grasp_pose)
            robot.lin(action)
            robot.gripper.close()
            robot.lin(pre_grasp_pose)
            env.remove_coordinate_frames()

        # Task-spezifische Aufräumaktion
        task.clean(env)

        # Hier nun der manuelle Abbau: 
        # z.B. Objekte an [0.3, -0.3, 0.0] ablegen
        #drop_position = [0.4, 0.0, 0.0]
        #cleanup_plan_1(bullet_client, object_ids, robot, drop_position, rotation)

    # Session beenden
    cv2.destroyAllWindows()
    with stdout_redirected():
        bullet_client.disconnect()

if __name__ == "__main__":
    main()
