import carla
from carla import VehicleLightState as vls

import os
import queue
import random
import threading

import cv2
import math
import numpy as np

import logging
PRELIMINARY_FILTER_DISTANCE = 100

MAX_RENDER_DEPTH_IN_METERS = 50
MIN_VISIBLE_VERTICES_FOR_RENDER = 3
MAX_OUT_VERTICES_FOR_RENDER = 5

TOWN_MAP = 1
LABEL_INDEX = 0

IMAGE_PATH = os.path.join("data", "gt", "carla", "carla_2d_box_train", "image")
LABEL_FILE = os.path.join(
    "data", "gt", "carla", "carla_2d_box_train", "label", f"{LABEL_INDEX:04d}.txt")

# Create a new directory if it does not exist
if not os.path.exists(os.path.dirname(LABEL_FILE)):
    os.makedirs(os.path.dirname(LABEL_FILE))
if not os.path.exists(IMAGE_PATH):
    os.makedirs(os.path.dirname(IMAGE_PATH))

try:
    f_label = open(LABEL_FILE, "w+")
except OSError:
    print("Could not open file:", LABEL_FILE)
    exit(1)


def calculate_occlusion_stats(vertices_pos2d, depth_image, image_h, image_w):
    """ 作用：筛选bbox八个顶点中实际可见的点 """
    num_visible_vertices = 0
    num_vertices_outside_camera = 0

    for y_2d, x_2d, vertex_depth in vertices_pos2d:
        # 点在可见范围中，并且没有超出图片范围
        if MAX_RENDER_DEPTH_IN_METERS > vertex_depth > 0 and point_in_canvas((y_2d, x_2d), image_h, image_w):
            is_occluded = point_is_occluded(
                (y_2d, x_2d, vertex_depth), depth_image)
            if not is_occluded:
                num_visible_vertices += 1
        else:
            num_vertices_outside_camera += 1
    return num_visible_vertices, num_vertices_outside_camera


def proj_to_camera(pos_vector, extrinsic_mat):
    """ 作用：将点的world坐标转换到相机坐标系中 """
    # inv求逆矩阵
    transformed_3d_pos = np.dot(np.linalg.inv(extrinsic_mat), pos_vector)
    return transformed_3d_pos


def midpoint_from_agent_location(location, extrinsic_mat):
    """ 将agent在世界坐标系中的中心点转换到相机坐标系下 """
    midpoint_vector = np.array([
        [location.x],  # [[X,
        [location.y],  # Y,
        [location.z],  # Z,
        [1.0]  # 1.0]]
    ])
    transformed_3d_midpoint = proj_to_camera(midpoint_vector, extrinsic_mat)
    return transformed_3d_midpoint


def get_relative_rotation_y(agent_rotation, obj_rotation):
    """ 返回actor和camera在rotation yaw的相对角度 """

    rot_agent = agent_rotation.yaw
    rot_car = obj_rotation.yaw
    return degrees_to_radians(rot_agent - rot_car)


def degrees_to_radians(degrees):
    return degrees * math.pi / 180


def transform_points(transform, points):
    """ 作用：将三维点坐标转换到指定坐标系下 """
    # 转置
    points = points.transpose()
    # [[X0..,Xn],[Y0..,Yn],[Z0..,Zn],[1,..1]]  (4,8)
    points = np.append(points, np.ones((1, points.shape[1])), axis=0)
    # transform.get_matrix() 获取当前坐标系向相对坐标系的旋转矩阵
    points = np.mat(transform.get_matrix()) * points
    # 返回前三行
    return points[0:3].transpose()


def vertices_from_extension(ext):
    """ 以自身为原点的八个点的坐标 """
    return np.array([
        [ext.x, ext.y, ext.z],      # Top left front
        [- ext.x, ext.y, ext.z],    # Top left back
        [ext.x, - ext.y, ext.z],    # Top right front
        [- ext.x, - ext.y, ext.z],  # Top right back
        [ext.x, ext.y, - ext.z],    # Bottom left front
        [- ext.x, ext.y, - ext.z],  # Bottom left back
        [ext.x, - ext.y, - ext.z],  # Bottom right front
        [- ext.x, - ext.y, - ext.z]  # Bottom right back
    ])


def bbox_2d_from_agent(intrinsic_mat, extrinsic_mat, obj_bbox, obj_transform, obj_tp):
    bbox = vertices_from_extension(obj_bbox.extent)
    if obj_tp == 1:
        bbox_transform = carla.Transform(obj_bbox.location, obj_bbox.rotation)
        bbox = transform_points(bbox_transform, bbox)
    else:
        box_location = carla.Location(obj_bbox.location.x-obj_transform.location.x,
                                      obj_bbox.location.y-obj_transform.location.y,
                                      obj_bbox.location.z-obj_transform.location.z)
        box_rotation = obj_bbox.rotation
        bbox_transform = carla.Transform(box_location, box_rotation)
        bbox = transform_points(bbox_transform, bbox)
    # 获取bbox在世界坐标系下的点的坐标
    bbox = transform_points(obj_transform, bbox)
    # 将世界坐标系下的bbox八个点转换到二维图片中
    vertices_pos2d = vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat)
    return vertices_pos2d


def vertices_to_2d_coords(bbox, intrinsic_mat, extrinsic_mat):
    """将bbox在世界坐标系中的点投影到该相机获取二维图片的坐标和点的深度"""
    vertices_pos2d = []
    for vertex in bbox:
        # 获取点在world坐标系中的向量
        pos_vector = vertex_to_world_vector(vertex)
        # 将点的world坐标转换到相机坐标系中
        transformed_3d_pos = proj_to_camera(pos_vector, extrinsic_mat)
        # 将点的相机坐标转换为二维图片的坐标
        pos2d = proj_to_2d(transformed_3d_pos, intrinsic_mat)
        # 点实际的深度
        vertex_depth = pos2d[2]
        # 点在图片中的坐标
        x_2d, y_2d = pos2d[0], pos2d[1]
        vertices_pos2d.append((y_2d, x_2d, vertex_depth))
    return vertices_pos2d


def vertex_to_world_vector(vertex):
    """ 以carla世界向量（X，Y，Z，1）返回顶点的坐标 （4,1）"""
    return np.array([
        [vertex[0, 0]],  # [[X,
        [vertex[0, 1]],  # Y,
        [vertex[0, 2]],  # Z,
        [1.0]  # 1.0]]
    ])


def proj_to_2d(camera_pos_vector, intrinsic_mat):
    """将相机坐标系下的点的3d坐标投影到图片上"""
    cords_x_y_z = camera_pos_vector[:3, :]
    cords_y_minus_z_x = np.concatenate(
        [cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
    pos2d = np.dot(intrinsic_mat, cords_y_minus_z_x)
    # normalize the 2D points
    pos2d = np.array([
        pos2d[0] / pos2d[2],
        pos2d[1] / pos2d[2],
        pos2d[2]
    ])
    return pos2d


def spawn_npc(client, nbr_vehicles, nbr_walkers, vehicles_list, all_walkers_id):
    world = client.get_world()

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)

    # traffic_manager.set_hybrid_physics_mode(True)
    # traffic_manager.set_random_device_seed(args.seed)

    traffic_manager.set_synchronous_mode(True)
    synchronous_master = True

    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

    safe = True
    if safe:
        blueprints = [x for x in blueprints if int(
            x.get_attribute('number_of_wheels')) == 4]
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]

    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)
    print("Number of spawn points : ", number_of_spawn_points)

    if nbr_vehicles <= number_of_spawn_points:
        random.shuffle(spawn_points)
    elif nbr_vehicles > number_of_spawn_points:
        msg = 'requested %d vehicles, but could only find %d spawn points'
        logging.warning(msg, nbr_vehicles, number_of_spawn_points)
        nbr_vehicles = number_of_spawn_points

    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor

    # --------------
    # Spawn vehicles
    # --------------
    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= nbr_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(
                blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(
                blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # prepare the light state of the cars to spawn
        light_state = vls.NONE
        car_lights_on = False
        if car_lights_on:
            light_state = vls.Position | vls.LowBeam | vls.LowBeam

        # spawn the cars and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform)
                     .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                     .then(SetVehicleLightState(FutureActor, light_state)))

    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)

    # -------------
    # Spawn Walkers
    # -------------
    # some settings
    walkers_list = []
    percentagePedestriansRunning = 0.0            # how many pedestrians will run
    # how many pedestrians will walk through the road
    percentagePedestriansCrossing = 0.0
    # 1. take all the random locations to spawn
    spawn_points = []
    all_loc = []
    i = 0
    while i < nbr_walkers:
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if ((loc != None) and not (loc in all_loc)):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
            all_loc.append(loc)
            i = i + 1
    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute(
                    'speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute(
                    'speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp,
                     carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put altogether the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_walkers_id.append(walkers_list[i]["con"])
        all_walkers_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_walkers_id)

    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    world.tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_walkers_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(
            world.get_random_location_from_navigation())
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    print('Spawned %d vehicles and %d walkers' %
          (len(vehicles_list), len(walkers_list)))

    # example of how to use parameters
    traffic_manager.global_percentage_speed_difference(30.0)


def follow(transform, world):
    # Transforme carla.Location(x,y,z) from sensor to world frame
    rot = transform.rotation
    rot.pitch = -25
    world.get_spectator().set_transform(carla.Transform(
        transform.transform(carla.Location(x=-15, y=0, z=5)), rot))

# Part 1


client = carla.Client('localhost', 2000)
client.set_timeout(100.0)

# Load the map
print("Map Town0"+str(TOWN_MAP))
world = client.load_world("Town0"+str(TOWN_MAP))

world = client.get_world()

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 1 / 1000.0
settings.no_rendering_mode = False
world.apply_settings(settings)

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

# Part 2


def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)

    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal

    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = np.array(
        [point_camera[1], -point_camera[2], point_camera[0]]).T

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)

    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img


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


# Spawn vehicles and walkers
spawn_npc(client, 50, 0,
          [], [])

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

env_object_ids = []

for obj in (car_objects + truck_objects + bus_objects):
    env_object_ids.append(obj.id)

# Disable all static vehicles
world.enable_environment_objects(env_object_ids, False)

edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
         [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]


def point_is_occluded(point, depth_map):
    """ Checks whether or not the four pixels directly around the given point has less depth than the given vertex depth
        If True, this means that the point is occluded.
    """
    x, y, vertex_depth = map(int, point)

    from itertools import product
    neigbours = product((1, -1), repeat=2)

    is_occluded = []
    for dy, dx in neigbours:
        # If the point is on the boundary
        if x == (depth_map.shape[1] - 1) or y == (depth_map.shape[0] - 1):
            is_occluded.append(True)
        # If the depth map says the pixel is closer to the camera than the actual vertex
        elif depth_map[y + dy, x + dx] < vertex_depth:
            is_occluded.append(True)
        else:
            is_occluded.append(False)
    # Only say point is occluded if all four neighbours are closer to camera than vertex
    return all(is_occluded)


def point_in_canvas(pos, img_h, img_w):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False


def get_vanishing_point(p1, p2, p3, p4):

    k1 = (p4[1] - p3[1]) / (p4[0] - p3[0])
    k2 = (p2[1] - p1[1]) / (p2[0] - p1[0])

    vp_x = (k1 * p3[0] - k2 * p1[0] + p1[1] - p3[1]) / (k1 - k2)
    vp_y = k1 * (vp_x - p3[0]) + p3[1]

    return [vp_x, vp_y]


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


edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
         [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

# spawn camera
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

depth_image_queue = queue.Queue()
depth_camera.listen(
    lambda image: depth_camera_callback(image, depth_image_queue))

# Get the world to camera matrix
world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
extrinsic = camera.get_transform().get_matrix()

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
while True:
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

            kitti_labels = []
            for npc in world.get_actors().filter('*vehicle*'):

                # Filter out the ego vehicle
                if npc.id != vehicle.id:

                    bb = npc.bounding_box
                    dist = npc.get_transform().location.distance(vehicle.get_transform().location)

                    # Filter for the vehicles within 50m
                    if dist < PRELIMINARY_FILTER_DISTANCE:
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
                                        (x - p_in_canvas[0]) + p_in_canvas[1]

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

                            vertices_pos2d = bbox_2d_from_agent(
                                K, extrinsic, npc.bounding_box, npc.get_transform(), 1)
                            num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(
                                vertices_pos2d, depth_map, image_h, image_w)

                            truncated = num_vertices_outside_camera / 8
                            if num_visible_vertices >= 6:
                                occluded = 0
                            elif num_visible_vertices >= 4:
                                occluded = 1
                            else:
                                occluded = 2

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
                                                 0,
                                                 x_min, y_min, x_max, y_max, z_min, z_max,
                                                 height, width, length,
                                                 loc_x, loc_y, loc_z,
                                                 rotation_y,
                                                 ])

            for label in kitti_labels:
                id, type, tuncated, occluded, alpha, x_min, y_min, x_max, y_max, z_min, z_max, height, width, length, loc_x, loc_y, loc_z, rotation_y = label

                # Exclude very small bounding boxes
                if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                    if point_in_canvas((x_min, y_min), image_h, image_w) and point_in_canvas((x_max, y_max), image_h, image_w):
                        if occluded <= 2:
                            cv2.line(img, (int(x_min), int(y_min)), (int(
                                x_max), int(y_min)), (0, 0, 255, 255), 1)
                            cv2.line(img, (int(x_min), int(y_max)), (int(
                                x_max), int(y_max)), (0, 0, 255, 255), 1)
                            cv2.line(img, (int(x_min), int(y_min)), (int(
                                x_min), int(y_max)), (0, 0, 255, 255), 1)
                            cv2.line(img, (int(x_max), int(y_min)), (int(
                                x_max), int(y_max)), (0, 0, 255, 255), 1)
                f_label.write(
                    f"{frame_id} {id} Car {truncated} {occluded} {alpha} {x_min} {y_min} {x_max} {y_max} {height} {width} {length} {loc_x} {loc_y} {loc_z} {rotation_y}\n")

            # Save image data
            file_path = os.path.join(
                "data", "gt", "carla", "carla_2d_box_train", "image", f"{LABEL_INDEX:04d}", f"{frame_id:04d}.png")
            t_save = threading.Thread(target=image.save_to_disk,
                                      args=(file_path, carla.ColorConverter.Raw))
            t_save.start()
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
