import sys
import time
import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from hydra.utils import instantiate
import copy
import math
import os
import cv2
import random
from scipy.spatial.transform import Rotation as R

# Falls deine Projektstruktur anders ist, Pfade ggf. anpassen:
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Bullet & Transform
from bullet_env.util import setup_bullet_client
from transform.affine import Affine

# Pointcloud / Heightmap - Funktionen
from convert_dataset import create_pointcloud, transform_pointcloud, filter_workspace, create_heightmap

# Transporter-Netzwerk
from lib.transporter_network.model import TransporterNetworkAttention


# ---------------------------------------------------------------------------
# combine_base_and_z
# ---------------------------------------------------------------------------
def combine_base_and_z(dtheta_rad, debug=False):
    """
    Kombiniert die Basisrotation [180, 0, 0] (XYZ) mit einer z-Rotation (dtheta_rad).
    => final = z_rot * base_rot
    """
    base_rot = R.from_euler('XYZ', [180, 0, 0], degrees=True)
    z_rot    = R.from_euler('z', dtheta_rad, degrees=False)
    final    = z_rot * base_rot
    if debug:
        eul = final.as_euler('XYZ', degrees=True)
        print("[DEBUG combine_base_and_z] Orientation (XYZ):", eul)
    return final.as_quat()

# ---------------------------------------------------------------------------
# snap_quader_orientation_to_short_side
# ---------------------------------------------------------------------------
def snap_quader_orientation_to_short_side(dtheta_deg):
    """
    'Snappt' Quader-Ausrichtung an die kurze Seite. Beispiel-Logik:
     - short-Kandidaten: [90, 270], long-Kandidaten: [0, 180]
     - wenn wir näher an 'long' sind, verschieben um ±90 => 'short'.
    """
    dtheta_mod = dtheta_deg % 360.0
    short_candidates = [90.0, 270.0]
    long_candidates  = [0.0, 180.0]

    # Distanz zu short vs. long
    dist_short = min(abs(dtheta_mod - s) for s in short_candidates)
    dist_long  = min(abs(dtheta_mod - l) for l in long_candidates)

    if dist_long < dist_short:
        # => näher an 'long' => + oder - 90
        dist0   = abs(dtheta_mod - 0.0)
        dist180 = abs(dtheta_mod - 180.0)
        if dist0 < dist180:
            dtheta_deg += 0.0
        else:
            dtheta_deg -= 0.0

    # optional runden auf 10°
    dtheta_deg = round(dtheta_deg/10.0)*10.0
    return dtheta_deg

# ---------------------------------------------------------------------------
# Plan 1
# ---------------------------------------------------------------------------
def plan_1(bc, base_position, rotation_deg, cube_urdf, quader_urdf):
    """
    Plan 1:
      - 2 Würfel nebeneinander, offset=0.15
      - je ein Würfel obendrauf
      - Quader (10×5×5) oben drauf
    """
    cube_size = 5.0
    offset    = 0.15

    rad = math.radians(rotation_deg)
    cos_theta = math.cos(rad)
    sin_theta = math.sin(rad)

    def rotate(x, y):
        return (x*cos_theta - y*sin_theta, x*sin_theta + y*cos_theta)

    def qz(a_deg):
        a= math.radians(a_deg)
        return [0,0, math.sin(a/2), math.cos(a/2)]

    object_info= []

    # Untere Ebene
    positions_lower= [(0,0,0), (cube_size+offset,0,0)]
    for (lx,ly,lz) in positions_lower:
        rx, ry= rotate(lx,ly)
        spawn= [
            base_position[0]+ rx/100.0,
            base_position[1]+ ry/100.0,
            base_position[2]+ lz/100.0
        ]
        ori= qz(rotation_deg)
        uid= bc.loadURDF(cube_urdf, spawn, ori)
        object_info.append([uid, spawn, ori, "cube"])

    # Obere Ebene
    positions_upper= [(0,0,cube_size+offset),
                      (cube_size+ offset,0,cube_size+offset)]
    for (ux,uy,uz) in positions_upper:
        rx, ry= rotate(ux,uy)
        spawn_u= [
            base_position[0]+ rx/100.0,
            base_position[1]+ ry/100.0,
            base_position[2]+ uz/100.0
        ]
        ori_u= qz(rotation_deg)
        uid_u= bc.loadURDF(cube_urdf, spawn_u, ori_u)
        object_info.append([uid_u, spawn_u, ori_u, "cube"])

    # Quader oben (10×5×5)
    quader_x= (cube_size+offset)/2.0
    quader_z= 2*(cube_size+ offset)
    rxQ, ryQ= rotate(quader_x, 0)
    spawnQ= [
        base_position[0]+ rxQ/100.0,
        base_position[1]+ ryQ/100.0,
        base_position[2]+ quader_z/100.0
    ]
    oriQ= qz(rotation_deg)
    uidQ= bc.loadURDF(quader_urdf, spawnQ, oriQ)
    object_info.append([uidQ, spawnQ, oriQ, "quader"])

    return object_info

# ---------------------------------------------------------------------------
# Plan 2
# ---------------------------------------------------------------------------
def plan_2(bc, base_position, rotation_deg, cube_urdf, quader_urdf):
    """
    Plan 2 => Torbogen
    """
    cube_size= 5.0
    offset= 0.1
    gap= 4.0

    rad= math.radians(rotation_deg)
    cos_t= math.cos(rad)
    sin_t= math.sin(rad)

    def rotate_xy(x,y):
        return (x*cos_t - y*sin_t, x*sin_t + y*cos_t)

    def qz(a_deg):
        a= math.radians(a_deg)
        return [0,0, math.sin(a/2), math.cos(a/2)]

    object_info= []

    left_bottom= (0,0,0)
    left_top   = (0,0, cube_size+ offset)
    right_bottom= (cube_size+ gap,0,0)
    right_top   = (cube_size+ gap,0,cube_size+ offset)

    quader_center_x= (0+ (cube_size+gap))/2.0
    quader_center_z= (cube_size+ offset)+ 2.5
    top_quader= (quader_center_x, 0, quader_center_z)

    positions= [left_bottom, left_top, right_bottom, right_top, top_quader]
    for i,(px,py,pz) in enumerate(positions):
        rx, ry= rotate_xy(px,py)
        spawn= [
            base_position[0]+ rx/100.0,
            base_position[1]+ ry/100.0,
            base_position[2]+ pz/100.0
        ]
        ori= qz(rotation_deg)
        if i<4:
            uid= bc.loadURDF(cube_urdf, spawn, ori)
            typ= "cube"
        else:
            uid= bc.loadURDF(quader_urdf, spawn, ori)
            typ= "quader"
        object_info.append([uid, spawn, ori, typ])

    return object_info

# ---------------------------------------------------------------------------
# Plan 3
# ---------------------------------------------------------------------------
def plan_3(bc, base_position, rotation_deg, cube_urdf, quader_urdf):
    """
    Ebene1: 2 Quader (10×5×5) => +90
    Ebene2: bridging => rotation_deg
    Ebene3: 2 Würfel => rotation_deg
    """
    quader_len= 10.0
    gap= 2.0
    offset_z= 0.1
    cube_size=5.0

    def qz(a_deg):
        a= math.radians(a_deg)
        return [0,0, math.sin(a/2), math.cos(a/2)]

    # rotation
    rad= math.radians(rotation_deg)
    cos_t= math.cos(rad)
    sin_t= math.sin(rad)

    def rotate_xy(x,y):
        return (x*cos_t - y*sin_t, x*sin_t + y*cos_t)

    object_info= []

    # Ebene1 => +90
    A_pos= (0,0,0)
    B_pos= (quader_len+ gap, 0,0)
    for pos in [A_pos,B_pos]:
        rx, ry= rotate_xy(pos[0], pos[1])
        spawn_p= [
            base_position[0]+ rx/100.0,
            base_position[1]+ ry/100.0,
            base_position[2]+ pos[2]/100.0
        ]
        ori= qz(rotation_deg + 90)
        uid= bc.loadURDF(quader_urdf, spawn_p, ori)
        object_info.append([uid, spawn_p, ori, "quader"])

    # Ebene2 => bridging
    top_z2= 5.0+ offset_z
    midx= (A_pos[0]+ B_pos[0])/2.0
    C_pos= (midx, 0, top_z2)
    rxC, ryC= rotate_xy(C_pos[0], C_pos[1])
    spawnC= [
        base_position[0]+ rxC/100.0,
        base_position[1]+ ryC/100.0,
        base_position[2]+ C_pos[2]/100.0
    ]
    oriC= qz(rotation_deg)
    uidC= bc.loadURDF(quader_urdf, spawnC, oriC)
    object_info.append([uidC, spawnC, oriC, "quader"])

    # Ebene3 => 2 Würfel
    D_pos= (C_pos[0], C_pos[1], 10.1)
    E_pos= (C_pos[0], C_pos[1], 15.2)
    for pos in [D_pos, E_pos]:
        rx, ry= rotate_xy(pos[0], pos[1])
        spawnD= [
            base_position[0]+ rx/100.0,
            base_position[1]+ ry/100.0,
            base_position[2]+ pos[2]/100.0
        ]
        oriD= qz(rotation_deg)
        uidD= bc.loadURDF(cube_urdf, spawnD, oriD)
        object_info.append([uidD, spawnD, oriD, "cube"])

    return object_info

# ---------------------------------------------------------------------------
# choose_random_plan
# ---------------------------------------------------------------------------
def choose_random_plan(bc, base_position, rotation_deg, cube_urdf, quader_urdf):
    plans= [plan_1, plan_2, plan_3]
    chosen= random.choice(plans)
    print(f"[INFO] Gewählter Plan: {chosen.__name__}")
    return chosen(bc, base_position, rotation_deg, cube_urdf, quader_urdf)

# ---------------------------------------------------------------------------
# run_inference_loop
# ---------------------------------------------------------------------------
def run_inference_loop(cfg, bc, env, robot, task, cf, model, place_xyz, max_repeats=5):
    """
    Kernschleife: 
      - Heightmap+Colormap
      - Inferenz
      - Quader => snap short side
      - neighbor check => +90
      - final => pick/place
      - repeated pixel => return False
      - empty => return True
    """
    neighbor_cm  = 1.0
    z_threshold  = 0.005
    attn_minval  = 0.0005

    place_x, place_y, place_z= place_xyz
    last_pxpy= None
    repeat_count= 0

    while True:
        # Warte => Simulation
        for _ in range(100):
            bc.stepSimulation()
            time.sleep(0.01)

        # Kameradaten
        robot.home()
        all_points, all_colors= [], []
        for cam in cf.cameras:
            obs= cam.get_observation()
            rgb, depth= obs['rgb'], obs['depth']
            intr, extr= obs['intrinsics'], obs['extrinsics']
            pts, cols= create_pointcloud(rgb, depth, intr)
            pts= transform_pointcloud(pts, extr)
            all_points.append(pts)
            all_colors.append(cols)
        if not all_points:
            print("[WARN] => Keine Kameradaten => continue.")
            continue
        all_points= np.concatenate(all_points, axis=0)
        all_colors= np.concatenate(all_colors, axis=0)
        all_points, all_colors= filter_workspace(all_points, all_colors, cfg.workspace_bounds)
        heightmap, colormap= create_heightmap(all_points, all_colors, [256,256], cfg.workspace_bounds)

        if np.max(heightmap)<0.02:
            print("[INFO] => Keine Objekte >2cm => fertig => True.")
            return True

        # Netz
        combined_input= np.dstack([colormap, heightmap])
        x_tf= tf.cast(np.expand_dims(combined_input,0), tf.float32)
        #cv2.imshow(combined_input)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        attn,yaws= model.infer(x_tf, training=False)
        attn_np= attn[0].numpy()
        yaws_np= yaws[0].numpy()

        idx= np.argmax(attn_np)
        py, px= np.unravel_index(idx, attn_np.shape)
        attn_val= attn_np[py, px]

        # repeat check
        pxpy= (px,py)
        if pxpy== last_pxpy:
            repeat_count+= 1
            print(f"[WARN] => Selber Pixel => count={repeat_count}")
        else:
            repeat_count=0
        last_pxpy= pxpy

        if repeat_count> max_repeats:
            print("[ERROR] => repeated pixel => return False.")
            return False

        if attn_val< attn_minval:
            print(f"[INFO] => attn={attn_val:.4f} < {attn_minval} => skip.")
            continue

        # Yaw
        yaw_idx= np.argmax(yaws_np)
        rot_steps= np.linspace(0, np.pi, model.n_orentation_bins)
        dtheta_rad= rot_steps[yaw_idx]
        dtheta_deg= math.degrees(dtheta_rad)

        # Farb
        pred_color= colormap[py, px].astype(np.float32)
        dist_cube=   np.linalg.norm(pred_color- np.array([255,0,0],dtype=np.float32))
        dist_quader= np.linalg.norm(pred_color- np.array([0,255,0],dtype=np.float32))
        block_type= "cube"
        if dist_quader< dist_cube:
            block_type= "quader"
            old_deg= dtheta_deg
            dtheta_deg= snap_quader_orientation_to_short_side(dtheta_deg)
            print(f"[INFO] => Quader snap {old_deg:.1f}°-> {dtheta_deg:.1f}°")

        # pixel->welt
        x_bounds,y_bounds,_= cfg.workspace_bounds
        H,W= attn_np.shape
        px_size= (x_bounds[1]- x_bounds[0])/ float(W)
        py_size= (y_bounds[1]- y_bounds[0])/ float(H)
        wx= x_bounds[0]+ px*px_size
        wy= y_bounds[0]+ py*py_size
        z_val= float(heightmap[py, px])

        # Visualization
        hm_min, hm_max= np.min(heightmap), np.max(heightmap)
        if hm_max> hm_min:
            hm_norm= (heightmap- hm_min)/(hm_max- hm_min)
        else:
            hm_norm= heightmap*0
        hm_disp= (hm_norm*255).astype(np.uint8)
        hm_disp_color= cv2.cvtColor(hm_disp, cv2.COLOR_GRAY2BGR)
        cv2.circle(hm_disp_color,(px,py),5, (0,0,255), -1)
        cv2.imshow("Heightmap Prediction", hm_disp_color)
        cv2.waitKey(1)

        # neighbor-check
        px_radius= int(neighbor_cm/ px_size)
        conflict_left= False
        conflict_right= False
        for dx in range(1, px_radius+1):
            if px-dx>=0:
                if abs(float(heightmap[py, px-dx]) - z_val)< z_threshold:
                    conflict_left= True
                    break
        for dx in range(1, px_radius+1):
            if px+dx< W:
                if abs(float(heightmap[py, px+dx]) - z_val)< z_threshold:
                    conflict_right= True
                    break
        if conflict_left and conflict_right:
            old_ = dtheta_deg
            dtheta_deg+= 90
            print(f"[INFO] => both-sides conflict => {old_:.1f}° -> {dtheta_deg:.1f}°")

        # Runden
        dtheta_deg= round(dtheta_deg/10)*10
        dtheta_rad= math.radians(dtheta_deg)

        if z_val<0.02:
            print("[INFO] => z<2cm => skip iteration.")
            continue
        pick_z= z_val- 0.025
        if pick_z<0:
            print("[WARN] => pick_z<0 => skip iteration.")
            continue

        q= combine_base_and_z(dtheta_rad)
        print(f"[INFO] => px=({px},{py}), attn={attn_val:.4f}, block={block_type},"
              f" (wx={wx:.3f}, wy={wy:.3f}, z={z_val:.3f}), yaw={dtheta_deg:.1f}°")

        # PICK
        
        pre_pick= Affine([wx, wy, pick_z+0.15], q)
        pick_aff= Affine([wx, wy, pick_z], q)
        robot.ptp(pre_pick)
        bc.stepSimulation()
        robot.lin(pick_aff)
        bc.stepSimulation()
        robot.gripper.close()
        bc.stepSimulation()
        robot.lin(pre_pick)
        bc.stepSimulation()

        # PLACE
        pre_place= Affine([place_x, place_y, place_z+0.15], q)
        place_aff= Affine([place_x, place_y, place_z], q)
        robot.lin(pre_place)
        bc.stepSimulation()
        robot.lin(place_aff)
        bc.stepSimulation()
        robot.gripper.open()
        bc.stepSimulation()
        robot.ptp(pre_place)
        bc.stepSimulation()

        place_x+= 0.15
        print("[INFO] => Next iteration...\n")
        time.sleep(1.0)

# ---------------------------------------------------------------------------
def plan_wrapper(cfg):
    """
    Eine äußere Schleife:
      - Startet PyBullet/Env
      - Lädt Modell
      - Führe z.B. 3 Versuche durch => spawn random plan => run_inference_loop
        => bei repeated Pixel => Abbruch => next attempt
        => am Ende bc.disconnect
    """
    bc= setup_bullet_client(cfg.render)
    env= instantiate(cfg.env, bullet_client=bc)
    robot= instantiate(cfg.robot, bullet_client=bc)

    t_bounds= copy.deepcopy(robot.workspace_bounds)
    t_bounds[2,1]= t_bounds[2,0]
    task_factory= instantiate(cfg.task_factory, t_bounds=t_bounds)
    oracle= instantiate(cfg.oracle)
    cf= instantiate(cfg.camera_factory, bullet_client=bc, t_center=np.mean(t_bounds, axis=1))

    # Modell
    model= TransporterNetworkAttention(64,18)
    checkpoint_prefix= "/home/jovyan/data/models/tn_model/tn_model"
    if not (os.path.exists(checkpoint_prefix+".index") and os.path.exists(checkpoint_prefix+".data-00000-of-00001")):
        raise FileNotFoundError("Checkpoints fehlen!")
    ckpt= tf.train.Checkpoint(model=model)
    ckpt.restore(checkpoint_prefix).expect_partial()
    print("[INFO] Transporter-Netzwerk geladen.")

    # Mehrere Versuche
    for attempt_i in range(1):
        print(f"\n=== Plan-Versuch {attempt_i+1} ===")

        spawn_x= round(np.random.uniform(0.65,0.75),3)
        spawn_y= round(np.random.uniform(-0.1,0.1),3)
        spawn_pos= [spawn_x, spawn_y, 0.0]
        rot_deg= int(np.random.choice(range(0,360,10)))
        print(f"[INFO] => spawn={spawn_pos}, rotation={rot_deg}")

        cube_urdf   = "/home/jovyan/data/assets/objects/cube/object.urdf"
        quader_urdf = "/home/jovyan/data/assets/objects/quader/object.urdf"
        objects_info= choose_random_plan(bc, spawn_pos, rot_deg, cube_urdf, quader_urdf)

        # init
        robot.home()
        robot.gripper.open()

        # Task
        object_specs= []
        for uid,pos,ori,typ in objects_info:
            object_specs.append({
                "object_type": typ,
                "object_id": uid,
                "position": pos,
                "orientation": ori
            })
        task= task_factory.create_task_from_specs(object_specs)
        oracle.solve(task)

        # run loop
        res= run_inference_loop(cfg, bc, env, robot, task, cf, model, [0.2, -0.35, 0.02], max_repeats=2)
        if res:
            print("[INFO] => run_inference_loop => OK/fertig.")
        else:
            print("[INFO] => run_inference_loop => repeated pixel => NEUER Versuch.")
        task.clean(env)
        print("[INFO] => Plan gesäubert => next.\n")

    print("[END] => Alle Versuche fertig, PyBullet beenden.")
    bc.disconnect()
    cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="config", config_name="tn_train_data")
def test_transporter_network(cfg: DictConfig) -> None:
    plan_wrapper(cfg)


if __name__=="__main__":
    test_transporter_network()
