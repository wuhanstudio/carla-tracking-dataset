import os
import argparse
import threading

import cv2
import numpy as np

import carla
from utils.world import KittiWorld

DATA_FOLDER = os.path.join("data", "gt", "carla", "carla_2d_box_train")

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

    print(f"Start Recording: Town0{args.town}_{args.index:04d}.log")
    client.start_recorder(f"Town0{args.town}_{args.index:04d}.log")

    kitti_world = KittiWorld(client)
    kitti_world.init()

    frame_id = 0

    # Main Loop
    while (frame_id < args.frame):
        try:
            image, kitti_labels = kitti_world.spin()

            if image is not None:
                # The RGB image
                img = np.reshape(np.copy(image.raw_data),
                                    (image.height, image.width, 4))

                # Save image data
                file_path = os.path.join(IMAGE_PATH, f"{frame_id:04d}.png")
                t_save = threading.Thread(target=image.save_to_disk,
                                            args=(file_path, carla.ColorConverter.Raw))
                t_save.start()
                print(f"Saving frame to {file_path}")

                # Save kitti labels
                for label in kitti_labels:
                    id, type, truncated, occluded, alpha, x_min, y_min, x_max, y_max, height, width, length, loc_x, loc_y, loc_z, rotation_y = label

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
                    f_label.flush()

                frame_id = frame_id + 1

                cv2.imshow('2D Bounding Boxes', img)

                if cv2.waitKey(1) == ord('q'):
                    f_label.close()
                    kitti_world.clear()
                    break
    
        except KeyboardInterrupt as e:
            f_label.close()
            kitti_world.clear()
            break

    print(f"Stop recording: Town0{args.town}_{args.index:04d}.log")
    client.stop_recorder()

    cv2.destroyAllWindows()
