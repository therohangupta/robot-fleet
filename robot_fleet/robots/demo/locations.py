try:
    from tf.transformations import euler_from_quaternion
    print("Using ROS 1")
except ImportError:
    from tf_transformations import euler_from_quaternion
    print("Using ROS 2")

locations = {
    "locobot": {
        "hallway": (-1.6102466262,1.62310173, euler_from_quaternion([0,0,-0.0609088, 0.99814333])[2]),
        "kitchen": (-2.636469, 0.79343536, euler_from_quaternion([0,0, -0.6770396889, 0.7359465])[2]),
        "living_room": (-1.587962628, -1.82796542, euler_from_quaternion([0,0,-0.0940912, 0.99556357])[2])

    },
    "hsr": {
        "hallway": (0.94559615,1.78532713, euler_from_quaternion([0,0, -0.51878346, 0.854905679])[2]),
        "kitchen": (-0.019480102, 2.59393566, euler_from_quaternion([0,0, 0.7138592, 0.7002891])[2]),
        "living_room": (-2.7940035786, 1.431508288, euler_from_quaternion([0,0,-0.503146307, 0.8642012459])[2])
    }
}