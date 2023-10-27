import os
import queue
import random
import argparse
import threading

import cv2
import math
import numpy as np

import carla
from utils.projection import *
from utils.world import follow, spawn_npc

MAX_RENDER_DEPTH_IN_METERS = 100
DATA_FOLDER = os.path.join("data", "gt", "carla", "carla_2d_box_train")

# Remember the edge pairs
edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
         [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]


def depth_camera_callback(image, depth_image_queue):
    # image.convert(carla.ColorConverter.LogarithmicDepth)

    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # RGBA format

    array = array[:, :, :3]     # Take only RGB
    array = array[:, :, ::-1]   # BGR

    array = array.astype(np.float32)  # 2ms

    gray_depth = ((array[:, :, 0]
                   + array[:, :, 1] * 256.0
                   + array[:, :, 2] * 256.0 * 256.0) / ((256.0 * 256.0 * 256.0) - 1)
                  )  # 2.5ms
    gray_depth = 1000 * gray_depth

    depth_image_queue.put(gray_depth)


def clear():
    settings = world.get_settings()
    settings.synchronous_mode = False  # Disables synchronous mode
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    camera.stop()
    depth_camera.stop()

    for npc in world.get_actors().filter('*vehicle*'):
        if npc:
            npc.destroy()

    print("Vehicles Destroyed.")


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Carla dataset generator')
    parser.add_argument(
        '--town',
        type=int,
        default=1,
        choices=range(0, 8),
        metavar="[0-7]",
        help='Map index: Town 1, 2, 3, ..., 7'
    )
    parser.add_argument(
        '--frame',
        type=int,
        default=300,
        help='Number of frames to collect (10 FPS).'
    )
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='You may collect the data several times, and wish to save them in different folders: 0, 1, 2, ...'
    )
    args = parser.parse_args()

    IMAGE_PATH = os.path.join(
        DATA_FOLDER, f"Town0{args.town}", "image", f"{args.index:04d}")
    LABEL_FILE = os.path.join(
        DATA_FOLDER, f"Town0{args.town}", "label", f"{args.index:04d}.txt")

    # Create a new directory if it does not exist
    if not os.path.exists(os.path.dirname(LABEL_FILE)):
        os.makedirs(os.path.dirname(LABEL_FILE))
    if not os.path.exists(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)
    try:
        f_label = open(LABEL_FILE, "w+")
    except OSError:
        print("Could not open file:", LABEL_FILE)
        exit(1)

    client = carla.Client('localhost', 2000)
    client.set_timeout(100.0)

    # Load the map
    print("Loading Map Town0" + str(args.town))
    world = client.load_world("Town0" + str(args.town))
    world = client.get_world()

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1 / 1000.0
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    # Clear existing NPC first
    for npc in world.get_actors().filter('*vehicle*'):
        if npc:
            npc.destroy()

    # Get the world spectator
    spectator = world.get_spectator()

    # Get the map spawn points
    spawn_points = world.get_map().get_spawn_points()

    # spawn vehicle
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('color', '228, 239, 241')
    vehicle_bp.set_attribute('role_name', 'KITTI')
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

    # Spawn vehicles and walkers
    spawn_npc(client, 50, 0)

    # Wait for KITTI to stop
    start = world.get_snapshot().timestamp.elapsed_seconds
    print("Waiting for KITTI to stop ...")
    while world.get_snapshot().timestamp.elapsed_seconds-start < 2.0:
        world.tick()
    print("KITTI stopped")

    # Retrieve all the objects of the level
    car_objects = world.get_environment_objects(
        carla.CityObjectLabel.Car)  # doesn't have filter by type yet
    truck_objects = world.get_environment_objects(
        carla.CityObjectLabel.Truck)  # doesn't have filter by type yet
    bus_objects = world.get_environment_objects(
        carla.CityObjectLabel.Bus)  # doesn't have filter by type yet

    # Disable all static vehicles
    env_object_ids = []
    for obj in (car_objects + truck_objects + bus_objects):
        env_object_ids.append(obj.id)
    world.enable_environment_objects(env_object_ids, False)

    # Spawn camera
    camera_init_trans = carla.Transform(carla.Location(
        x=0.30, y=0, z=1.70), carla.Rotation(pitch=0, yaw=0, roll=0))

    # Create a RGB camera
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1392')
    camera_bp.set_attribute('image_size_y', '1024')
    # 72 degrees # Always fov on width even if width is different than height
    camera_bp.set_attribute('fov', '72')
    camera_bp.set_attribute('enable_postprocess_effects', 'True')
    camera_bp.set_attribute('sensor_tick', '0.10')  # 10Hz camera
    camera_bp.set_attribute('gamma', '2.2')
    camera_bp.set_attribute('motion_blur_intensity', '0')
    camera_bp.set_attribute('motion_blur_max_distortion', '0')
    camera_bp.set_attribute('motion_blur_min_object_screen_size', '0')
    camera_bp.set_attribute('shutter_speed', '1000')  # 1 ms shutter_speed
    camera_bp.set_attribute('lens_k', '0')
    camera_bp.set_attribute('lens_kcube', '0')
    camera_bp.set_attribute('lens_x_size', '0')
    camera_bp.set_attribute('lens_y_size', '0')
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

    # Create a depth camera
    depth_camera_bp = world.get_blueprint_library().find('sensor.camera.depth')
    depth_camera_bp.set_attribute('image_size_x', '1392')
    depth_camera_bp.set_attribute('image_size_y', '1024')
    # 72 degrees # Always fov on width even if width is different than height
    depth_camera_bp.set_attribute('fov', '72')
    depth_camera_bp.set_attribute('sensor_tick', '0.10')  # 10Hz camera
    depth_camera = world.spawn_actor(
        depth_camera_bp, camera_init_trans, attach_to=vehicle)

    def camera_callback(sensor_data, sensor_queue):
        sensor_queue.put(sensor_data)

    # Create a queue to store and retrieve the sensor data
    image_queue = queue.Queue()
    camera.listen(lambda data: camera_callback(data, image_queue))

    # Create a depth camera for occlusion calculation
    depth_image_queue = queue.Queue()
    depth_camera.listen(
        lambda image: depth_camera_callback(image, depth_image_queue))

    # Get the attributes from the camera
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix(image_w, image_h, fov)
    K_b = build_projection_matrix(image_w, image_h, fov, is_behind_camera=True)

    vehicle.set_autopilot(True)

    # Pass to the next simulator frame to spawn sensors and to retrieve first data
    world.tick()

    ts_tmp = 0
    initial_ts = world.get_snapshot().timestamp.elapsed_seconds

    frame_id = 0

    # Main Loop
    while (frame_id < args.frame):
        try:
            # Retrieve and reshape the image
            if not image_queue.empty():
                image = image_queue.get()
                depth_map = depth_image_queue.get()

                ts = image.timestamp - initial_ts
                if (ts - ts_tmp > 0.11) or (ts - ts_tmp) < 0:  # check for 10Hz camera acquisition
                    print("[Error in timestamp] Camera: previous_ts %f -> ts %f" %
                          (ts_tmp, ts))
                    break
                ts_tmp = ts

                img = np.reshape(np.copy(image.raw_data),
                                 (image.height, image.width, 4))

                # Get the camera matrix
                world_2_camera = np.array(
                    camera.get_transform().get_inverse_matrix())

                extrinsic = camera.get_transform().get_matrix()

                kitti_labels = []
                for npc in world.get_actors().filter('*vehicle*'):

                    # Filter out the ego vehicle
                    if npc.id != vehicle.id:

                        bb = npc.bounding_box
                        dist = npc.get_transform().location.distance(vehicle.get_transform().location)

                        # Filter for the vehicles within 100m
                        if dist < MAX_RENDER_DEPTH_IN_METERS:
                            # Calculate the dot product between the forward vector
                            # of the vehicle and the vector between the vehicle
                            # and the other vehicle. We threshold this dot product
                            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA

                            forward_vec = vehicle.get_transform().get_forward_vector()
                            ray = npc.get_transform().location - vehicle.get_transform().location

                            if forward_vec.dot(ray) > 0:

                                verts = [v for v in bb.get_world_vertices(
                                    npc.get_transform())]

                                points_image = []

                                for vert in verts:
                                    ray0 = vert - camera.get_transform().location
                                    cam_forward_vec = camera.get_transform().get_forward_vector()

                                    if (cam_forward_vec.dot(ray0) > 0):
                                        p = get_image_point(
                                            vert, K, world_2_camera)
                                    else:
                                        p = get_image_point(
                                            vert, K_b, world_2_camera)

                                    points_image.append(p)

                                x_min, x_max = 10000, -10000
                                y_min, y_max = 10000, -10000
                                z_min, z_max = 10000, -10000

                                for edge in edges:
                                    p1 = points_image[edge[0]]
                                    p2 = points_image[edge[1]]

                                    p1_in_canvas = point_in_canvas(
                                        p1, image_h, image_w)
                                    p2_in_canvas = point_in_canvas(
                                        p2, image_h, image_w)

                                    # Both points are out of the canvas
                                    if not p1_in_canvas and not p2_in_canvas:
                                        continue

                                    # Draw 3D Bounding Boxes
                                    # cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)

                                    # Draw 2D Bounding Boxes
                                    p1_temp, p2_temp = (p1.copy(), p2.copy())

                                    # One of the point is out of the canvas
                                    if not (p1_in_canvas and p2_in_canvas):
                                        p = [0, 0]

                                        # Find the intersection of the edge with the window border
                                        p_in_canvas, p_not_in_canvas = (
                                            p1, p2) if p1_in_canvas else (p2, p1)
                                        k = (
                                            p_not_in_canvas[1] - p_in_canvas[1]) / (p_not_in_canvas[0] - p_in_canvas[0])

                                        x = np.clip(
                                            p_not_in_canvas[0], 0, image.width)
                                        y = k * \
                                            (x - p_in_canvas[0]
                                             ) + p_in_canvas[1]

                                        if y >= image.height:
                                            p[0] = (image.height - p_in_canvas[1]
                                                    ) / k + p_in_canvas[0]
                                            p[1] = image.height - 1
                                        elif y <= 0:
                                            p[0] = (0 - p_in_canvas[1]) / \
                                                k + p_in_canvas[0]
                                            p[1] = 0
                                        else:
                                            p[0] = image.width - \
                                                1 if x == image.width else 0
                                            p[1] = y

                                        p1_temp, p2_temp = (p, p_in_canvas)

                                    # Find the rightmost vertex
                                    x_max = p1_temp[0] if p1_temp[0] > x_max else x_max
                                    x_max = p2_temp[0] if p2_temp[0] > x_max else x_max

                                    # Find the leftmost vertex
                                    x_min = p1_temp[0] if p1_temp[0] < x_min else x_min
                                    x_min = p2_temp[0] if p2_temp[0] < x_min else x_min

                                    # Find the highest vertex
                                    y_max = p1_temp[1] if p1_temp[1] > y_max else y_max
                                    y_max = p2_temp[1] if p2_temp[1] > y_max else y_max

                                    # Find the lowest vertex
                                    y_min = p1_temp[1] if p1_temp[1] < y_min else y_min
                                    y_min = p2_temp[1] if p2_temp[1] < y_min else y_min

                                    # No depth information means the point is on the boundary
                                    if len(p1_temp) == 3:
                                        z_max = p1_temp[2] if p1_temp[2] > z_max else z_max
                                        z_min = p1_temp[2] if p1_temp[2] < z_min else z_min

                                    if len(p2_temp) == 3:
                                        z_max = p2_temp[2] if p2_temp[2] > z_max else z_max
                                        z_min = p2_temp[2] if p2_temp[2] < z_min else z_min

                                # Exclude very small bounding boxes
                                if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                                    if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):

                                        vertices_pos2d = bbox_2d_from_agent(
                                            K, extrinsic, npc.bounding_box, npc.get_transform(), 1)
                                        num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(
                                            vertices_pos2d, depth_map, MAX_RENDER_DEPTH_IN_METERS)

                                        # Use 8 3D vertices to calculate occlusion
                                        # if num_visible_vertices >= 6:
                                        #     occluded = 0
                                        # elif num_visible_vertices >= 4:
                                        #     occluded = 1
                                        # else:
                                        #     occluded = 2

                                        # Use 4 2D vertices to calculate occlusion
                                        o1 = point_is_occluded((y_min, x_min, z_min), depth_map)
                                        o2 = point_is_occluded((y_min, x_min, z_max), depth_map)
                                        o3 = point_is_occluded((y_max, x_max, z_min), depth_map)
                                        o4 = point_is_occluded((y_max, x_max, z_max), depth_map)

                                        # Not all points are occluded
                                        if (o1 + o2 + o3 + o4 < 2):
                                            occluded = 0
                                        elif (o1 + o2 + o3 + o4 < 3):
                                            occluded = 1
                                        else:
                                            occluded = 2

                                        truncated = num_vertices_outside_camera / 8

                                        cos_alpha = forward_vec.dot(
                                            ray) / np.sqrt(ray.squared_length())
                                        if cos_alpha > 1 or cos_alpha < -1:
                                            if np.allclose(cos_alpha, 1):
                                                cos_alpha = 1
                                            elif np.allclose(cos_alpha, -1):
                                                cos_alpha = -1
                                            else:
                                                print("Error: Invalid ALpha")
                                        alpha = np.arccos(cos_alpha)

                                        rotation_y = get_relative_rotation_y(
                                            npc.get_transform().rotation, npc.get_transform().rotation) % math.pi

                                        # Bbox extent consists of x,y and z.
                                        # The bbox extent is by Carla set as
                                        # x: length of vehicle (driving direction)
                                        # y: to the right of the vehicle
                                        # z: up (direction of car roof)
                                        # However, Kitti expects height, width and length (z, y, x):
                                        # Since Carla gives us bbox extent, which is a half-box, multiply all by two
                                        bbox_extent = npc.bounding_box.extent
                                        height, width, length = bbox_extent.z, bbox_extent.x, bbox_extent.y
                                        loc_x, loc_y, loc_z = [float(x) for x in midpoint_from_agent_location(
                                            npc.get_transform().location, extrinsic)][0:3]

                                        kitti_labels.append([npc.id,
                                                            "Car",
                                                             truncated,
                                                             occluded,
                                                             alpha,
                                                             x_min, y_min, x_max, y_max, z_min, z_max,
                                                             height, width, length,
                                                             loc_x, loc_y, loc_z,
                                                             rotation_y,
                                                             ])

                for label in kitti_labels:
                    id, type, tuncated, occluded, alpha, x_min, y_min, x_max, y_max, z_min, z_max, height, width, length, loc_x, loc_y, loc_z, rotation_y = label

                    # BGR - Visible: Blue, Partially Visible: Yellow, Invisible: Red
                    colors = [(255, 0, 0), (0, 255, 255), (0, 0, 255)]

                    cv2.line(img, (int(x_min), int(y_min)),
                             (int(x_max), int(y_min)), colors[occluded], 1)
                    cv2.line(img, (int(x_min), int(y_max)),
                             (int(x_max), int(y_max)), colors[occluded], 1)
                    cv2.line(img, (int(x_min), int(y_min)),
                             (int(x_min), int(y_max)), colors[occluded], 1)
                    cv2.line(img, (int(x_max), int(y_min)),
                             (int(x_max), int(y_max)), colors[occluded], 1)

                    f_label.write(
                        f"{frame_id} {id} Car {truncated} {occluded} {alpha:.6f} {x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f} {height:.6f} {width:.6f} {length:.6f} {loc_x:.6f} {loc_y:.6f} {loc_z:.6f} {rotation_y:.6f}\n")

                # Save image data
                file_path = os.path.join(IMAGE_PATH, f"{frame_id:04d}.png")
                t_save = threading.Thread(target=image.save_to_disk,
                                          args=(file_path, carla.ColorConverter.Raw))
                t_save.start()
                f_label.flush()
                print(f"Saving frame to {file_path}")

                frame_id = frame_id + 1

                cv2.imshow('2D Bounding Boxes', img)

                if cv2.waitKey(1) == ord('q'):
                    f_label.close()
                    clear()
                    break

            # Move the spectator to the top of the vehicle
            transform = carla.Transform(vehicle.get_transform().transform(
                carla.Location(x=-4, z=50)), carla.Rotation(yaw=-180, pitch=-90))
            spectator.set_transform(transform)

            world.tick()

        except KeyboardInterrupt as e:
            f_label.close()
            clear()
            break

    cv2.destroyAllWindows()
