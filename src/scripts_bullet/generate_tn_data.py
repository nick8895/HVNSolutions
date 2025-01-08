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

# Falls die folgenden Module in deinem Projekt vorhanden sind, 
# bleibe bei diesen Imports. Andernfalls anpassen oder entfernen.
from image_util import draw_pose
from data_util import store_data_grasp


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

    # Anzahl der zu generierenden Szenen
    for i in range(cfg.n_scenes):
        # Roboter initialisieren
        robot.home()
        robot.gripper.open()

        # Task erstellen (aber NICHT die zufälligen Objekte spawnen!)
        task = task_factory.create_task()
        # task.setup(env)  # AUSKOMMENTIERT, um das zufällige Spawnen zu verhindern

        # Feste Objekte manuell in die Szene laden
        object_path = "/home/jovyan/data/assets/objects/cube/object.urdf"      # Anpassen: Pfad zu deinem URDF
        spawn_position = [0.7, 0.0, 0.0]        # Anpassen: (x, y, z)
        spawn_orientation = [0.0, 0.0, 0.0, 1]  # Quaternion (x, y, z, w)
        object_id = bullet_client.loadURDF(
            object_path,
            spawn_position,
            spawn_orientation
        )

        # Hole ggf. Task-Infos (falls dein Oracle das braucht)
        task_info = task.get_info()

        # Die Greif-Pose über das Oracle bestimmen
        pose = oracle.solve(task)

        # Kamerabilder aufnehmen
        observations = [camera.get_observation() for camera in camera_factory.cameras]

        # Daten speichern (optional)
        if cfg.store_dataset:
            store_data_grasp(i, task_info, observations, pose, cfg.dataset_directory)

        # Debug-/Visualisierungsmodus
        if cfg.debug:
            # RGB- und Depth-Bilder anzeigen
            image_copy = copy.deepcopy(observations[0]['rgb'])
            depth_copy = copy.deepcopy(observations[0]['depth'])

            # Greifpose im RGB-Bild visualisieren
            draw_pose(observations[0]['extrinsics'], pose, observations[0]['intrinsics'], image_copy)
            cv2.imshow('rgb', image_copy)

            # Depth-Bild reskalieren zur Anzeige
            depth_copy = depth_copy / 2.0
            cv2.imshow('depth', depth_copy)

            key_pressed = cv2.waitKey(0)
            if key_pressed == ord('q'):
                # Schleife abbrechen
                break

            # Greifaktion demonstrieren
            env.spawn_coordinate_frame(pose)
            action = Affine.from_matrix(pose)
            pre_grasp_offset = Affine([0, 0, -0.1])
            pre_grasp_pose = action * pre_grasp_offset

            robot.ptp(pre_grasp_pose)
            robot.lin(action)
            robot.gripper.close()
            robot.lin(pre_grasp_pose)
            env.remove_coordinate_frames()

        # Objekt und Task aufräumen
        # -> 'task.clean(env)' entfernt normalerweise vom Task erstellte Objekte;
        #    du kannst diese Zeile drin lassen, damit ggf. andere Ressourcen gereinigt werden.
        task.clean(env)

        # Das manuell geladene Objekt entfernen, 
        # um die nächste Szene "leer" zu starten (optional).
        bullet_client.removeBody(object_id)

    # PyBullet-Session beenden
    cv2.destroyAllWindows()
    with stdout_redirected():
        bullet_client.disconnect()


if __name__ == "__main__":
    main()
