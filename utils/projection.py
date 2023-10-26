import carla

import math
import numpy as np

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


def point_is_occluded(point, depth_map):
    """ Checks whether or not the four pixels directly around the given point has less depth than the given vertex depth
        If True, this means that the point is occluded.
    """
    y, x, vertex_depth = map(int, point)

    from itertools import product
    neigbours = product((1, -1), repeat=2)

    is_occluded = []
    for dy, dx in neigbours:
        # If the point is on the boundary
        if x == (depth_map.shape[1] - 1) or y == (depth_map.shape[0] - 1):
            dy = dx = 0
        # If the depth map says the pixel is closer to the camera than the actual vertex
        if depth_map[y + dy, x + dx] < vertex_depth:
            is_occluded.append(True)
        else:
            is_occluded.append(False)
    # Only say point is occluded if all four neighbours are closer to camera than vertex
    return all(is_occluded)

def calculate_occlusion_stats(vertices_pos2d, depth_image, MAX_RENDER_DEPTH_IN_METERS):
    """ 作用：筛选bbox八个顶点中实际可见的点 """
    num_visible_vertices = 0
    num_vertices_outside_camera = 0

    image_h, image_w = depth_image.shape

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
