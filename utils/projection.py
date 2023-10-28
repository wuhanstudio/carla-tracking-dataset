import math
import numpy as np

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

def get_2d_box_from_3d_edges(points_2d, edges, image_h, image_w):
    x_min, x_max = 10000, -10000
    y_min, y_max = 10000, -10000

    for edge in edges:
        p1 = points_2d[edge[0]]
        p2 = points_2d[edge[1]]

        p1_in_canvas = point_in_canvas(p1, image_h, image_w)
        p2_in_canvas = point_in_canvas(p2, image_h, image_w)

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
                p_not_in_canvas[0], 0, image_w)
            y = k * \
                (x - p_in_canvas[0]
                ) + p_in_canvas[1]

            if y >= image_h:
                p[0] = (image_h - p_in_canvas[1]
                        ) / k + p_in_canvas[0]
                p[1] = image_h - 1
            elif y <= 0:
                p[0] = (0 - p_in_canvas[1]) / \
                    k + p_in_canvas[0]
                p[1] = 0
            else:
                p[0] = image_w - \
                    1 if x == image_w else 0
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

    return x_min, x_max, y_min, y_max

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


def calculate_occlusion_stats(vertices_pos2d, depth_image, MAX_RENDER_DEPTH_IN_METERS):
    """ 作用：筛选bbox八个顶点中实际可见的点 """
    num_visible_vertices = 0
    num_vertices_outside_camera = 0

    image_h, image_w = depth_image.shape

    for x_2d, y_2d, vertex_depth in vertices_pos2d:
        # 点在可见范围中，并且没有超出图片范围
        if MAX_RENDER_DEPTH_IN_METERS > vertex_depth > 0 and point_in_canvas((x_2d, y_2d), image_h, image_w):
            is_occluded = point_is_occluded(
                (x_2d, y_2d, vertex_depth), depth_image)
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


def degrees_to_radians(degrees):
    return degrees * math.pi / 180


def get_relative_rotation_y(agent_rotation, obj_rotation):
    """ 返回actor和camera在rotation yaw的相对角度 """

    rot_agent = agent_rotation.yaw
    rot_car = obj_rotation.yaw
    return degrees_to_radians(rot_agent - rot_car)
