#!/usr/bin/env python3
import math
import time
import rospy
import numpy as np

from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from dynamic_reconfigure.client import Client
from geometry_msgs.msg import Twist

from task2navpoints import task2goals
from ocr_detector import ocr_detector


# === Callback Functions ===
def amcl_callback(msg):
    global current_pose
    current_pose = msg.pose.pose

def costmap_callback(msg):
    global latest_costmap
    latest_costmap = msg

def global_costmap_callback(msg):
    global global_costmap_info
    global_costmap_info = msg.info

def image_callback(msg):
    global latest_image
    try:
        latest_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except Exception as e:
        print("Image processing failed:", e)

def depth_callback(msg):
    global latest_depth
    try:
        latest_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    except Exception as e:
        print("Depth image processing failed:", e)


# === Utility Functions ===
def make_pose(x, y, yaw=0.0):
    '''
    Input: x, y, yaw
    Output: goal pose in move_base format
    '''
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.z = math.sin(yaw / 2)
    pose.pose.orientation.w = math.cos(yaw / 2)
    return pose

def make_amcl_initial_pose(x, y, yaw=0.0, cov_xy=1.0, cov_yaw_deg=15):
    """
    Create initial pose message for /initialpose
    """
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = "map"
    msg.header.stamp = rospy.Time.now()
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.orientation.z = math.sin(yaw / 2)
    msg.pose.pose.orientation.w = math.cos(yaw / 2)

    cov_yaw_rad = math.radians(cov_yaw_deg)
    msg.pose.covariance[0] = cov_xy ** 2       # variance in x
    msg.pose.covariance[7] = cov_xy ** 2       # variance in y
    msg.pose.covariance[35] = cov_yaw_rad ** 2 # variance in yaw

    return msg

def get_yaw_from_quat(q):
    """
    Convert quaternion to yaw angle in radians
    """
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def is_goal_reached(goal, current):
    """
    Check if current pose has reached the goal
    """
    if goal is None or current is None:
        return False
    dx = goal.pose.position.x - current.position.x
    dy = goal.pose.position.y - current.position.y
    return math.hypot(dx, dy) < 0.5


def init_global_map_from_info(info):
    """
    Initialize global map for scanning boxes area
    """
    global global_map, global_info
    resolution = info.resolution
    global_width = info.width
    global_height = info.height
    global_map = np.full((global_height, global_width), 255, dtype=np.uint8)
    global_info = {
        'resolution': resolution,
        'origin_x': info.origin.position.x,
        'origin_y': info.origin.position.y,
        'width': global_width,
        'height': global_height
    }

def integrate_local_into_global(local):
    """
    Integrate local costmap into global map
    """
    global global_map, global_info
    resolution = global_info['resolution']
    global_origin_x = global_info['origin_x']
    global_origin_y = global_info['origin_y']

    local_data = np.array(local.data).reshape((local.info.height, local.info.width))
    local_origin_x = local.info.origin.position.x
    local_origin_y = local.info.origin.position.y

    offset_x = int((local_origin_x - global_origin_x) / resolution)
    offset_y = int((local_origin_y - global_origin_y) / resolution)

    for i in range(local.info.height):
        for j in range(local.info.width):
            val = local_data[i, j]
            if val != -1:
                gx = offset_y + i
                gy = offset_x + j
                if 0 <= gx < global_map.shape[0] and 0 <= gy < global_map.shape[1]:
                    global_map[gx, gy] = val

def save_global_map(filename="merged_local_costmap.png"):
    """
    Save the merged costmap as an image
    """
    img = global_map.copy()
    img[img == -1] = 205
    img[img == 0] = 255
    img[img == 100] = 0
    img = np.flipud(img) 
    PILImage.fromarray(img.astype(np.uint8)).save(filename)

def adjust_yaw_based_depth(latest_depth, atol=0.1):
    """
    Adjust robot yaw based on depth image to better face the object
    """
    depth_array = np.array(latest_depth, dtype=np.float32)
    img_h, img_w = depth_array.shape

    masked_depth = np.where((depth_array > 0) & (~np.isnan(depth_array)), depth_array, np.inf)
    min_val = np.min(masked_depth)

    candidates = np.argwhere(np.isclose(masked_depth, min_val, atol))
    center = np.array([img_h / 2, img_w / 2])
    distances = [np.linalg.norm(p - center) for p in candidates]
    min_idx = candidates[np.argmin(distances)]

    dx = (min_idx[1] - img_w // 2)
    horizontal_fov_deg = 60.0
    horizontal_fov_rad = math.radians(horizontal_fov_deg)
    angle_offset = (dx / (img_w / 2)) * (horizontal_fov_rad / 2)

    adjusted_yaw = get_yaw_from_quat(current_pose.orientation) + angle_offset
    adjusted_pose = make_pose(
        current_pose.position.x,
        current_pose.position.y,
        yaw=adjusted_yaw / 2
    )
    return adjusted_pose

def open_bridge():
    """
    Trigger bridge to open
    """
    pub = rospy.Publisher('/cmd_open_bridge', Bool, queue_size=1)
    rospy.sleep(1)
    pub.publish(True)
    rospy.loginfo("Published True to /cmd_open_bridge")


# === Global State Variables ===
state = 1
current_pose = None
next_goal = None
goal_reached_flag = False

bridge = CvBridge()

latest_costmap = None
global_map = None
global_info = None
global_costmap_info = None

number_count = {}
latest_image = None
latest_depth = None


rospy.init_node("task_node")
rate = rospy.Rate(5)

# Subscribers
rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, amcl_callback)
rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, costmap_callback)
rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, global_costmap_callback)
rospy.Subscriber("/front/rgb/image_raw", Image, image_callback)
rospy.Subscriber("/front/depth/image_raw", Image, depth_callback)

# Publisher
pub_goal = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)

# === Continue Main Task ===
while not rospy.is_shutdown():
    # State 1: Navigate to entrance of random area
    if state == 1:
        next_goal = make_pose(19.5, -22, yaw=math.radians(180))
        pub_goal.publish(next_goal)
        goal_reached_flag = is_goal_reached(next_goal, current_pose)
        if goal_reached_flag:
            state = 2
            goal_reached_flag = False
            print("State 1: Navigation to random area entrance complete.")

    # State 2: Explore fixed points within the random area
    elif state == 2:
        if latest_costmap and global_map is None and global_costmap_info:
            init_global_map_from_info(global_costmap_info)
        
        goals = [
            make_pose(10, -22, yaw=math.radians(90)),
            make_pose(10, -16, yaw=math.radians(0)),
            make_pose(19.5, -16, yaw=math.radians(90)),
            make_pose(19.5, -9, yaw=math.radians(180)),
            make_pose(10, -9, yaw=math.radians(90)),
            make_pose(10, -3, yaw=math.radians(0)),
            make_pose(19.5, -3, yaw=math.radians(-90))
        ]

        for goal in goals:
            pub_goal.publish(goal)
            rospy.sleep(1)
            while not rospy.is_shutdown():
                if latest_costmap:
                    integrate_local_into_global(latest_costmap)
                goal_reached_flag = is_goal_reached(goal, current_pose)
                if goal_reached_flag:
                    break
                rate.sleep()

        print("State 2: Random area map built and saved as merged_local_costmap.png")
        save_global_map()
        state = 3

    # State 3: Perform number recognition on each box
    elif state == 3:
        nav_boxes, nav_bridge, nav_final_goals = task2goals()
        visited = set()

        while len(visited) < len(nav_boxes) and not rospy.is_shutdown():
            if current_pose is None:
                rate.sleep()
                continue

            cur_x = current_pose.position.x
            cur_y = current_pose.position.y

            min_dist = float('inf')
            next_goal = None
            next_idx = None
            for i, (x, y, yaw_deg) in enumerate(nav_boxes):
                if i in visited:
                    continue
                dist = math.hypot(x - cur_x, y - cur_y)
                if dist < min_dist:
                    min_dist = dist
                    next_goal = make_pose(x, y, yaw=math.radians(yaw_deg))
                    next_idx = i

            if next_goal is not None:
                pub_goal.publish(next_goal)
                print(f"Navigating to point {next_idx + 1}/{len(nav_boxes)}: ({x:.2f}, {y:.2f})")

                while not rospy.is_shutdown():
                    if is_goal_reached(next_goal, current_pose):
                        print(f"Reached point {next_idx + 1}")
                        visited.add(next_idx)
                        rospy.sleep(2)

                        if latest_image is not None:
                            number = ocr_detector(latest_image)
                            if number:
                                number_count[number] = number_count.get(number, 0) + 1
                                print(f"Detected number: {number}, current count: {number_count}")
                            else:
                                print("No number detected.")
                        else:
                            print("No image received.")

                        break
                    rate.sleep()
        print("State 3: Completed number recognition for all boxes.")
        state = 4

    # State 4: Cross the bridge
    elif state == 4:
        next_goal = make_pose(nav_bridge[0] + 1, nav_bridge[1], yaw=math.radians(nav_bridge[2]))
        pub_goal.publish(next_goal)
        while not rospy.is_shutdown():
            if is_goal_reached(next_goal, current_pose):
                rospy.sleep(1)
                print("Reached bridge point 1.")
                break

        vel_msg = Twist()
        vel_msg.linear.x = 1
        vel_msg.angular.z = 0.0
        next_goal = make_pose(4, nav_bridge[1], nav_bridge[2])

        pub_velocity = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)
        rospy.sleep(1)
        pub_velocity.publish(vel_msg)
        open_bridge()

        rate_vel = rospy.Rate(10)
        while not rospy.is_shutdown():
            if is_goal_reached(next_goal, current_pose):
                break
            pub_velocity.publish(vel_msg)
            rate_vel.sleep()

        pub_init_amcl = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
        rospy.sleep(1)
        init_pose = make_amcl_initial_pose(x=4, y=nav_bridge[1], yaw=math.radians(nav_bridge[2]), cov_xy=1.5, cov_yaw_deg=20)
        pub_init_amcl.publish(init_pose)
        rospy.sleep(3)
        print("AMCL re-localization completed.")
        
        state = 5

    # State 5: Find the least frequent number
    elif state == 5:

        if number_count:
            target_number = min(number_count.items(), key=lambda x: x[1])[0]
            print(f"Target number is: {target_number}")
        else:
            print("No numbers detected. Skipping final navigation.")
            continue
        
        found = False
        for i, (x, y, yaw_deg) in enumerate(nav_final_goals):
            if found:
                break

            final_goal = make_pose(3, y, yaw=math.radians(yaw_deg))
            pub_goal.publish(final_goal)
            print(f"Navigating to final point {i+1}/4")

            while not rospy.is_shutdown():
                if is_goal_reached(final_goal, current_pose):
                    rospy.sleep(3)

                    if latest_image is not None:
                        number = ocr_detector(latest_image)
                        print(f"Detected number: {number}")
                        if number == target_number:
                            found = True
                            print("Target number matched. Moving forward 1 meter.")
                            forward_pose = make_pose(
                                current_pose.position.x + 1 * math.cos(math.radians(yaw_deg)),
                                current_pose.position.y + 1 * math.sin(math.radians(yaw_deg)),
                                yaw=math.radians(yaw_deg)
                            )
                            pub_goal.publish(forward_pose)
                            rospy.sleep(1)
                            break
                        else:
                            print("Not the target number. Continuing to next point.")
                    else:
                        print("No image received. Skipping.")

                    break

            rate.sleep()

        state = 6

    else:
        break 

    rate.sleep()
