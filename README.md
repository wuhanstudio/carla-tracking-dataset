# Carla Tracking Dataset

> A 2D trakcking dataset generated from CARLA simulator with sensors identical to the KITTI dataset.


To collect the data from Town 01, and save the data in folder `0000`:

```
$ python main.py --town 01 --index 0 --frame 300
```

The following bash script automatically collects a KITTI-like object detection dataset from Map 01-07 (3 runs for each map, 21 runs in total).

```
$ ./auto_collect_data.sh
```

To convert collected 2D images to videos:

```
$ ./auto_gen_mp4.sh
```


    data/
    └── gt
        └── carla
            └── carla_2d_box_train
                ├── Town01
                │   ├── image
                │   └── label
                ├── Town02
                │   ├── image
                │   └── label
                ├── Town03
                │   ├── image
                │   └── label
                ├── Town04
                │   ├── image
                │   └── label
                ├── Town05
                │   ├── image
                │   └── label
                ├── Town06
                │   ├── image
                │   └── label
                ├── Town07
                │   ├── image
                │   └── label
                ├── Town10
                │   ├── image
                │   └── label
                └── Town11
                    ├── image
                    └── label


![](dataset.png)
