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
from CoordinateManager import CoordinateManager

# Falls in deinem Projekt vorhanden; ansonsten ggf. anpassen oder entfernen
from image_util import draw_pose
from data_util import store_data_grasp

@hydra.main(version_base=None, config_path="config", config_name="tn_train_data")
def main(cfg: DictConfig) -> None:
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    # PyBullet starten (GUI oder Headless)
    bullet_client = setup_bullet_client(cfg.render)

    # Environment und Roboter instanziieren
    env = instantiate(cfg.env, bullet_client=bullet_client)
    robot = instantiate(cfg.robot, bullet_client=bullet_client)

    # Bounds für den Arbeitsraum (z auf Bodenebene setzen)
    t_bounds = copy.deepcopy(robot.workspace_bounds)
    t_bounds[2, 1] = t_bounds[2, 0]  # Gleiche min und max für z

    # Task-Factory und Oracle laden (Greiffunktion bleibt erhalten)
    task_factory = instantiate(cfg.task_factory, t_bounds=t_bounds)
    oracle = instantiate(cfg.oracle)

    # Kamera-Setup
    t_center = np.mean(t_bounds, axis=1)
    camera_factory = instantiate(cfg.camera_factory, bullet_client=bullet_client, t_center=t_center)
    manager = CoordinateManager()
    spawn_position = [0.6, 0.0, 0.0] 
    print("set coordinates")
    manager.set_coordinates(spawn_position)  # <--- Definierte Position (x, y)
    print("done")
    # Anzahl der Szenen
    for i in range(cfg.n_scenes):
        # Roboter zurücksetzen
        robot.home()
        robot.gripper.open()

        # Task anlegen – aber NICHT zufällige Objekte spawnen
       
        # task.setup(env)  # AUSKOMMENTIERT, damit keine zufälligen Objekte erzeugt werden

        

        # Feste, eigene URDF laden und spawnen
        object_path = "/home/jovyan/data/assets/objects/cube/object.urdf"      # <--- Pfad zum eigenen Objekt anpassen
        
        
        task = task_factory.create_task()
        task.setup(env)
        #spawn_position = [0.6, 0.0, 0.0]        # <--- Definierte Position (x, y, z)
        spawn_orientation = [0.0, 0.0, 0.0, 1]  # <--- Quaternion (x, y, z, w)
        object_id = bullet_client.loadURDF(
            object_path,
            spawn_position,
            spawn_orientation
        )

        # Task-Info holen (wird ggf. vom Oracle benötigt)
        task_info = task.get_info()

        # Oracle berechnet zufällige oder definierte Greifpose
        pose = oracle.solve(task)

        # Kameraaufnahmen sammeln
        observations = [camera.get_observation() for camera in camera_factory.cameras]

        # Daten speichern (falls in der Config aktiviert)
        if cfg.store_dataset:
            store_data_grasp(i, task_info, observations, pose, cfg.dataset_directory)

        # Debug-/Visualisierungsmodus
        if cfg.debug:
            # RGB- und Depth-Bild der ersten Kamera holen und kopieren
            image_copy = copy.deepcopy(observations[0]['rgb'])
            depth_copy = copy.deepcopy(observations[0]['depth'])

            # Greifpose im RGB-Bild darstellen
            draw_pose(
                observations[0]['extrinsics'],
                pose,
                observations[0]['intrinsics'],
                image_copy
            )

            # Anzeigen in OpenCV-Fenstern
            cv2.imshow('rgb', image_copy)
            cv2.imshow('depth', depth_copy / 2.0)  # Depth für Visualisierung reskalieren

            # Warten auf Tastendruck
            key_pressed = cv2.waitKey(0)
            if key_pressed == ord('q'):
                break

            # Koordinatensystem NICHT an der zufälligen Oracle-Pose,
            # sondern direkt an unserem gespawnten Objekt erzeugen.
            # Dafür bauen wir aus spawn_position (und ggf. -orientation) ein Affine.
            # Hier nur übersetzt als reine Translation (kein Rotationsteil),
            # falls du die Orientierung auch übertragen möchtest, musst du die Rotation
            # ebenfalls im Affine verarbeiten.
            object_pose = Affine(spawn_position)
            env.spawn_coordinate_frame(object_pose.matrix)
            
            # Roboterbewegung nach Oracle-Pose (bleibt erhalten)
            action = Affine.from_matrix(object_pose)
            pre_grasp_offset = Affine([0, 0, -0.1])
            pre_grasp_pose = action * pre_grasp_offset

            # Den Greifablauf demonstrieren
            robot.ptp(pre_grasp_pose)
            robot.lin(action)
            robot.gripper.close()
            robot.lin(pre_grasp_pose)

            # Koordinatensystem wieder entfernen
            env.remove_coordinate_frames()

        # Task aufräumen (löscht in der Regel Task-spezifische Objekte und Frames)
        task.clean(env)

        # Manuell gespawntes Objekt entfernen, damit die nächste Szene leer startet
        bullet_client.removeBody(object_id)

    # OpenCV-Fenster schließen (falls vorhanden)
    cv2.destroyAllWindows()

    # PyBullet-Session beenden
    with stdout_redirected():
        bullet_client.disconnect()


if __name__ == "__main__":
    main()
