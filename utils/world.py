import queue
import random
import logging

import math
import numpy as np

import carla
from carla import VehicleLightState as vls

from .projection import *

MAX_RENDER_DEPTH_IN_METERS = 100

class KittiWorld:
    def __init__(self, client) -> None:
        self.client = client
        self.world = client.get_world()

        # Get the world spectator
        self.spectator = self.world.get_spectator()

        # Set up the simulator in synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / 1000.0
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)

    def init(self):
        # Clear existing NPC
        self.clear_npc()

        # Clear static vehicles
        self.clear_static_vehicle()

        # Spawn the ego vehicle
        self.spawn_kitti()

        # Spawn vehicles and walkers
        self.spawn_npc(100, 0)

        # Wait for KITTI to stop
        start = self.world.get_snapshot().timestamp.elapsed_seconds
        print("Waiting for KITTI to stop ...")
        while self.world.get_snapshot().timestamp.elapsed_seconds-start < 2.0:
            self.world.tick()
        print("KITTI stopped")

        self.spawn_camera()

        self.vehicle.set_autopilot(True)

        # Pass to the next simulator frame to spawn sensors and to retrieve first data
        self.world.tick()

        self.__temp_ts__ = 0
        self.__init_ts__ = self.world.get_snapshot().timestamp.elapsed_seconds

    def spin(self):
        # Return the camera image and labels
        image = None
        kitti_labels = []

        while image is None:
            # Retrieve and reshape the image
            if not self.image_queue.empty():
                image = self.image_queue.get()
                depth_map = self.depth_image_queue.get()

                ts = image.timestamp - self.__init_ts__
                # check for 10Hz camera acquisition
                if (ts - self.__temp_ts__ > 0.11) or (ts - self.__temp_ts__) < 0:
                    print("[Error in timestamp] Camera: previous_ts %f -> ts %f" %
                        (self.__temp_ts__, ts))
                    return None, []

                self.__temp_ts__ = ts

                # Get the camera matrix
                world_2_camera = np.array(
                    self.camera.get_transform().get_inverse_matrix())

                extrinsic = self.camera.get_transform().get_matrix()

                for npc in self.world.get_actors().filter('*vehicle*'):

                    # Filter out the ego vehicle
                    if npc.id != self.vehicle.id:

                        bb = npc.bounding_box
                        dist = npc.get_transform().location.distance(
                            self.vehicle.get_transform().location)

                        # Filter for the vehicles within 100m
                        if dist < MAX_RENDER_DEPTH_IN_METERS:
                            # Calculate the dot product between the forward vector
                            # of the vehicle and the vector between the vehicle
                            # and the other vehicle. We threshold this dot product
                            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA

                            forward_vec = self.vehicle.get_transform().get_forward_vector()
                            ray = npc.get_transform().location - self.vehicle.get_transform().location

                            if forward_vec.dot(ray) > 0:

                                verts = [v for v in bb.get_world_vertices(
                                    npc.get_transform())]

                                points_2d = []

                                for vert in verts:
                                    ray0 = vert - self.camera.get_transform().location
                                    cam_forward_vec = self.camera.get_transform().get_forward_vector()

                                    if (cam_forward_vec.dot(ray0) > 0):
                                        p = get_image_point(
                                            vert, self.K, world_2_camera)
                                    else:
                                        p = get_image_point(
                                            vert, self.K_b, world_2_camera)

                                    points_2d.append(p)

                                # Remember the edge pairs
                                edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5],
                                        [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

                                x_min, x_max, y_min, y_max = get_2d_box_from_3d_edges(points_2d, edges, image.height, image.width)

                                # Exclude very small bounding boxes
                                if (y_max - y_min) * (x_max - x_min) > 100 and (x_max - x_min) > 20:
                                    if point_in_canvas((x_min, y_min), self.image_h, self.image_w) and point_in_canvas((x_max, y_max), self.image_h, self.image_w):

                                        num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(
                                            points_2d, depth_map, MAX_RENDER_DEPTH_IN_METERS)

                                        # Use 3D vertices to calculate occlusion
                                        if num_visible_vertices >= 6:
                                            occluded = 0
                                        elif num_visible_vertices >= 4:
                                            occluded = 1
                                        else:
                                            occluded = 2

                                        if not (occluded == 2 and dist > 50):
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
                                                                x_min, y_min, x_max, y_max,
                                                                height, width, length,
                                                                loc_x, loc_y, loc_z,
                                                                rotation_y,
                                                                ])

            self.follow()
            self.world.tick()

        return image, kitti_labels

    def spawn_kitti(self):
        # Get the map spawn points
        spawn_points = self.world.get_map().get_spawn_points()

        # spawn vehicle
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('color', '228, 239, 241')
        vehicle_bp.set_attribute('role_name', 'KITTI')
        self.vehicle = self.world.try_spawn_actor(
            vehicle_bp, random.choice(spawn_points))

    def spawn_npc(self, nbr_vehicles, nbr_walkers):
        vehicles_list = []
        all_walkers_id = []

        # self.world.set_weather(carla.WeatherParameters.ClearNoon)

        traffic_manager = self.client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)

        # traffic_manager.set_hybrid_physics_mode(True)
        # traffic_manager.set_random_device_seed(args.seed)

        traffic_manager.set_synchronous_mode(True)
        synchronous_master = True

        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        blueprintsWalkers = self.world.get_blueprint_library().filter('walker.pedestrian.*')

        safe = True
        if safe:
            blueprints = [x for x in blueprints if int(
                x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [
                x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [
                x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = self.world.get_map().get_spawn_points()
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

        for response in self.client.apply_batch_sync(batch, synchronous_master):
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
            loc = self.world.get_random_location_from_navigation()
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
        results = self.client.apply_batch_sync(batch, True)
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
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp,
                                    carla.Transform(), walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_walkers_id.append(walkers_list[i]["con"])
            all_walkers_id.append(walkers_list[i]["id"])
        all_actors = self.world.get_actors(all_walkers_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_walkers_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(
                self.world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('Spawned %d vehicles and %d walkers' %
              (len(vehicles_list), len(walkers_list)))

        # example of how to use parameters
        traffic_manager.global_percentage_speed_difference(30.0)

    def spawn_camera(self):
        # Spawn camera
        camera_init_trans = carla.Transform(carla.Location(
            x=0.30, y=0, z=1.70), carla.Rotation(pitch=0, yaw=0, roll=0))

        bp_lib = self.world.get_blueprint_library()

        # Create a RGB camera
        camera_bp = bp_lib.find('sensor.camera.rgb')

        camera_bp.set_attribute('image_size_x', '1392')
        camera_bp.set_attribute('image_size_y', '1024')

        # Windows Fix: https://github.com/carla-simulator/carla/issues/6085
        # camera_bp.set_attribute('image_size_x', '1280')
        # camera_bp.set_attribute('image_size_y', '640')

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

        self.camera = self.world.spawn_actor(
            camera_bp, camera_init_trans, attach_to=self.vehicle)

        # Create a depth camera
        depth_camera_bp = self.world.get_blueprint_library().find('sensor.camera.depth')

        depth_camera_bp.set_attribute('image_size_x', '1392')
        depth_camera_bp.set_attribute('image_size_y', '1024')

        # Windows Fix: https://github.com/carla-simulator/carla/issues/6085
        # depth_camera_bp.set_attribute('image_size_x', '1280')
        # depth_camera_bp.set_attribute('image_size_y', '640')

        # 72 degrees # Always fov on width even if width is different than height
        depth_camera_bp.set_attribute('fov', '72')
        depth_camera_bp.set_attribute('sensor_tick', '0.10')  # 10Hz camera

        self.depth_camera = self.world.spawn_actor(
            depth_camera_bp, camera_init_trans, attach_to=self.vehicle)

        def camera_callback(sensor_data, sensor_queue):
            sensor_queue.put(sensor_data)

        def depth_camera_callback(image, depth_image_queue):
            # image.convert(carla.ColorConverter.LogarithmicDepth)

            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(
                array, (image.height, image.width, 4))  # RGBA format

            array = array[:, :, :3]     # Take only RGB
            array = array[:, :, ::-1]   # BGR

            array = array.astype(np.float32)  # 2ms

            gray_depth = ((array[:, :, 0]
                           + array[:, :, 1] * 256.0
                           + array[:, :, 2] * 256.0 * 256.0) / ((256.0 * 256.0 * 256.0) - 1)
                          )  # 2.5ms
            gray_depth = 1000 * gray_depth

            depth_image_queue.put(gray_depth)

        # Create a queue to store and retrieve the sensor data
        self.image_queue = queue.Queue()
        self.camera.listen(
            lambda data: camera_callback(data, self.image_queue))

        # Create a depth camera for occlusion calculation
        self.depth_image_queue = queue.Queue()
        self.depth_camera.listen(
            lambda image: depth_camera_callback(image, self.depth_image_queue))

        # Get the attributes from the camera
        self.image_w = camera_bp.get_attribute("image_size_x").as_int()
        self.image_h = camera_bp.get_attribute("image_size_y").as_int()
        self.fov = camera_bp.get_attribute("fov").as_float()

        # Calculate the camera projection matrix to project from 3D -> 2D
        self.K = build_projection_matrix(self.image_w, self.image_h, self.fov)
        self.K_b = build_projection_matrix(
            self.image_w, self.image_h, self.fov, is_behind_camera=True)

    def follow(self):
        # Move the spectator to the top of the vehicle
        transform = carla.Transform(self.vehicle.get_transform().transform(
            carla.Location(x=-4, z=50)), carla.Rotation(yaw=-180, pitch=-90))
        self.spectator.set_transform(transform)

    def clear_npc(self):
        # Clear existing NPC first
        for npc in self.world.get_actors().filter('*vehicle*'):
            if npc:
                npc.destroy()

    def clear_static_vehicle(self):
        # Retrieve all the objects of the level
        car_objects = self.world.get_environment_objects(
            carla.CityObjectLabel.Car)  # doesn't have filter by type yet
        truck_objects = self.world.get_environment_objects(
            carla.CityObjectLabel.Truck)  # doesn't have filter by type yet
        bus_objects = self.world.get_environment_objects(
            carla.CityObjectLabel.Bus)  # doesn't have filter by type yet

        # Disable all static vehicles
        env_object_ids = []
        for obj in (car_objects + truck_objects + bus_objects):
            env_object_ids.append(obj.id)
        self.world.enable_environment_objects(env_object_ids, False)

    def clear(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False  # Disables synchronous mode
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        self.camera.stop()
        self.depth_camera.stop()

        for npc in self.world.get_actors().filter('*vehicle*'):
            if npc:
                npc.destroy()

        print("Vehicles Destroyed.")
