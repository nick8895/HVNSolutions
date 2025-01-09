import sys
import hydra
from loguru import logger
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
import copy
import cv2

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
        object_ids.append(obj_id)

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
    object_ids.append(quader_id)

    return object_ids

def pick_and_place_object(bullet_client, robot, object_id, place_position, rotation):
    """
    Greift das Objekt mit einer einfachen "Pick von oben"-Pose und legt es an der place_position ab.
    :param bullet_client: PyBullet-Client
    :param robot: Roboterobjekt (muss PTP, LIN, gripper.open(), .close() etc. unterstützen)
    :param object_id: Die ID des zu greifenden Objekts
    :param place_position: [x, y, z] in Metern, wo das Objekt abgelegt werden soll
    """

    # Hole aktuelle Objektposition und -orientierung
    obj_pos, obj_ori = bullet_client.getBasePositionAndOrientation(Affine(object_id))

    rad = math.radians(rotation)

    # Einfache "von oben" Greif-Orientierung (Identity / z-Achse nach unten)
    # Falls dein Roboter eine bestimmte Ausrichtung braucht, hier anpassen:
    pick_orientation = quaternion_from_rotation_z(rad)

    # Pre-/Post-Grasp-Offsets
    pre_grasp_offset_z = 0.20  # 20 cm über dem Objekt
    grasp_offset_z = 0.02      # 2 cm über Objektmittelpunkt
    place_offset_z = 0.02      # 2 cm über Ablageposition

    # Greif-Pose definieren (wir nehmen an: x, y unverändert, z + offset)
    pre_grasp_pose = Affine([
        obj_pos[0],
        obj_pos[1],
        obj_pos[2] + pre_grasp_offset_z
    ], pick_orientation)

    grasp_pose = Affine([
        obj_pos[0],
        obj_pos[1],
        obj_pos[2] + grasp_offset_z
    ], pick_orientation)

    # Platzier-Pose
    pre_place_pose = Affine([
        place_position[0],
        place_position[1],
        place_position[2] + pre_grasp_offset_z
    ], pick_orientation)

    place_pose = Affine([
        place_position[0],
        place_position[1],
        place_position[2] + place_offset_z
    ], pick_orientation)

    # 1) Zum Pre-Grasp-Punkt fahren
    robot.ptp(pre_grasp_pose)
    # 2) Gerade runter zum Grasp-Punkt
    robot.lin(grasp_pose)
    # 3) Greifer schließen
    robot.gripper.close()
    # 4) Objekt anheben (zurück zum Pre-Grasp)
    robot.lin(pre_grasp_pose)

    # 5) Über Ablage-Position fliegen
    robot.ptp(pre_place_pose)
    # 6) Gerade runter zur Ablage
    robot.lin(place_pose)
    # 7) Greifer öffnen
    robot.gripper.open()
    # 8) Wieder nach oben
    robot.lin(pre_place_pose)

    def quaternion_from_rotation_z(angle_rad):
        half_angle = angle_rad / 2.0
        return [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]

def cleanup_plan_1(bullet_client, object_ids, robot, drop_position, rotation):
    """
    Baut die in plan_1 generierten Objekte nacheinander (von oben nach unten) ab
    und legt sie an einer definierten Position ab.
    :param bullet_client: Instanz des PyBullet-Clients
    :param object_ids: Liste von Objekt-IDs (0..3 = Würfel, 4 = Quader) 
    :param robot: Roboterobjekt
    :param drop_position: [x, y, z] in Metern, wo die Objekte abgelegt werden
    """
    # Gehe in umgekehrter Reihenfolge (von oben nach unten) durch alle Objekte
    for obj_id in reversed(object_ids):
        pick_and_place_object(bullet_client, robot, obj_id, drop_position, rotation)

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

    for i in range(cfg.n_scenes):
        # Roboter initialisieren
        robot.home()
        robot.gripper.open()

        # Task erstellen (aber NICHT die zufälligen Objekte spawnen!)
        task = task_factory.create_task()
        # task.setup(env)  # Deaktiviert, damit keine zufälligen Objekte gespawnt werden

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
        drop_position = [0.4, 0.0, 0.0]
        cleanup_plan_1(bullet_client, object_ids, robot, drop_position, rotation)

    # Session beenden
    cv2.destroyAllWindows()
    with stdout_redirected():
        bullet_client.disconnect()

if __name__ == "__main__":
    main()
