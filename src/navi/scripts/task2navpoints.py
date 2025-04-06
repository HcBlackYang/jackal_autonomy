#!/usr/bin/env python3
import os
import math
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Extract object centroid from image
import numpy as np
import cv2

# Extract object centroids from image (with area-weighted center)
def find_object_centroids(gray_img, merge_threshold=23, min_area=25, split_area=384):
    binary = (gray_img < 255).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_centroids = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Determine if large areas need to be split
        if area > split_area and cx > 220:
            # Use mask to cover midline, try to split contour
            mask = np.zeros_like(gray_img, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mask[:, cx] = 0
            sub_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for sc in sub_contours:
                sub_area = cv2.contourArea(sc)
                if sub_area >= min_area:
                    M = cv2.moments(sc)
                    if M["m00"] > 0:
                        sx = int(M["m10"] / M["m00"])
                        sy = int(M["m01"] / M["m00"])
                        raw_centroids.append(((sx, sy), sub_area))
        else:
            raw_centroids.append(((cx, cy), area))

    # Area division
    mid_zone = [item for item in raw_centroids if 133 <= item[0][0] <= 217]
    left_zone = [item for item in raw_centroids if item[0][0] < 133]
    right_zone = [item for item in raw_centroids if item[0][0] > 217]

    merged = []

    # Add all centroids from left zone
    merged.extend([pt for pt, _ in left_zone])

    # Add a weighted average centroid for middle zone
    if mid_zone:
        total_area = sum([a for _, a in mid_zone])
        weighted_sum = np.sum([np.array(pt) * a for pt, a in mid_zone], axis=0)
        avg = tuple((weighted_sum / total_area).astype(int))
        merged.append(avg)

    # Merge nearby centroids in right zone
    right_points = [pt for pt, _ in right_zone]
    used = [False] * len(right_points)
    for i, c1 in enumerate(right_points):
        if used[i]:
            continue
        group = [c1]
        used[i] = True
        for j, c2 in enumerate(right_points):
            if not used[j] and np.linalg.norm(np.array(c1) - np.array(c2)) < merge_threshold:
                group.append(c2)
                used[j] = True
        merged.append(tuple(np.mean(group, axis=0).astype(int)))

    return merged


# Calculate navigation point coordinates
def find_navigation_points(new_img, centroids_img, box_half=16, check_dist=16, offset=0):
    nav_points = []
    nav_final_goals = []
    h, w = new_img.shape

    def is_near_obstacle(x, y, img, min_dist=4):
        for dy in range(-min_dist, min_dist + 1):
            for dx in range(-min_dist, min_dist + 1):
                xx = x + dx
                yy = y + dy
                if 0 <= xx < w and 0 <= yy < h:
                    if img[yy, xx] == 0:
                        return True
        return False

    for cx, cy in centroids_img:
        candidates = []

        if cx < 220:
            dx, dy = check_dist, 0
            px, py = cx + dx, cy + dy

            x1, x2 = max(px - box_half, 0), min(px + box_half, w)
            y1, y2 = max(py - box_half, 0), min(py + box_half, h)
            region = new_img[y1:y2, x1:x2]
            obstacle_pixels = np.sum(region == 0)
            best = ((dx, dy), (px, py), obstacle_pixels)
        else:
            directions = [
                (0, -check_dist),  # up
                (0, check_dist),   # down
                (-check_dist, 0),  # left
                (check_dist, 0)    # right
            ]

            for dx, dy in directions:
                px, py = cx + dx, cy + dy

                x1, x2 = max(px - box_half, 0), min(px + box_half, w)
                y1, y2 = max(py - box_half, 0), min(py + box_half, h)
                region = new_img[y1:y2, x1:x2]
                obstacle_pixels = np.sum(region == 0)
                candidates.append(((dx, dy), (px, py), obstacle_pixels))

            min_obstacles = min(candidates, key=lambda x: x[2])[2]
            best_candidates = [item for item in candidates if item[2] == min_obstacles]
            best = random.choice(best_candidates)

        (dx_best, dy_best), (px_best, py_best), _ = best

        nx = int(cx + dx_best + np.sign(dx_best) * offset)
        ny = int(cy + dy_best + np.sign(dy_best) * offset)

        nx = np.clip(nx, 0, w - 1)
        ny = np.clip(ny, 0, h - 1)

        max_attempts = 10

        # if 133 <= cx < 220:
        #     # First find a ny further from obstacles
        #     found = False
        #     for i in range(1, max_attempts + 1):
        #         ny_down = np.clip(ny + i, 0, h - 1)
        #         if not is_near_obstacle(nx, ny_down, new_img):
        #             ny = ny_down
        #             found = True
        #             break
        #         ny_up = np.clip(ny - i, 0, h - 1)
        #         if not is_near_obstacle(nx, ny_up, new_img):
        #             ny = ny_up
        #             found = True
        #             break

        #     # === New logic: find closest black pixels above and below (nx, ny) ===
        #     black_y_up = None
        #     black_y_down = None
        #     for i in range(1, h):
        #         if ny - i >= 0 and new_img[ny - i, nx] == 0 and black_y_up is None:
        #             black_y_up = ny - i
        #         if ny + i < h and new_img[ny + i, nx] == 0 and black_y_down is None:
        #             black_y_down = ny + i
        #         if black_y_up is not None and black_y_down is not None:
        #             break

        #     if black_y_up is not None and black_y_down is not None:
        #         ny = int((black_y_up + black_y_down) / 2)

        # else:
        #     if is_near_obstacle(nx, ny, new_img):
        #         possible_safe = []
        #         tx, ty = nx, ny
        #         for step in range(1, max_attempts + 1):
        #             tx += int(np.sign(dx_best))
        #             ty += int(np.sign(dy_best))
        #             tx = np.clip(tx, 0, w - 1)
        #             ty = np.clip(ty, 0, h - 1)
        #             if not is_near_obstacle(tx, ty, new_img):
        #                 possible_safe.append((tx, ty))
        #                 break
        #         if possible_safe:
        #             nx, ny = possible_safe[0]

        dx = cx - nx
        dy = cy - ny
        yaw = math.degrees(math.atan2(-dy, dx))

        point = (nx, ny, yaw)

        if cx < 133:
            nav_final_goals.append(point)
        else:
            nav_points.append(point)

    return nav_points, nav_final_goals


# Convert coordinates to map coordinate system
def get_centroids_map(img, centroids_img):
    resolution = 0.05
    origin_x = -2.244615
    origin_y = -24.2

    # Image dimensions
    img_height = img.shape[0]

    # Convert pixel centroids to map coordinates
    centroids_map = []
    if len(centroids_img[0]) == 2:
        for (cx, cy) in centroids_img:
            map_x = origin_x + resolution * cx
            map_y = origin_y + resolution * (img_height - cy)
            centroids_map.append((map_x, map_y))
    else:
        for (cx, cy, yaw) in centroids_img:
            map_x = origin_x + resolution * cx
            map_y = origin_y + resolution * (img_height - cy)
            centroids_map.append((map_x, map_y, yaw))

    return centroids_map

# Main function
def task2goals():
    image_path = "merged_local_costmap.png"

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Image does not exist: {image_path}")

    # Read image (grayscale mode)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_bg = np.where(img < 255, 0, 255).astype(np.uint8)

    # Create an all-white image (same size)
    new_img = np.full_like(img, 255)

    # Copy target area: Y 88-494, X 0-430
    new_img[88:494, 0:430] = img[88:494, 0:430]

    # Remove unknown area pixels
    new_img[new_img == 205] = 255
    new_img[new_img < 255] = 0

    # Extract centroids
    centroids_img = find_object_centroids(new_img, merge_threshold=20, min_area=25)
    navpoints_img, nav_final_goals_img = find_navigation_points(img_bg, centroids_img, box_half=16, check_dist=32, offset=4)

    # Convert to map coordinates
    centroids_map = get_centroids_map(new_img, centroids_img)
    navpoints_map = get_centroids_map(new_img, navpoints_img)


    # Extract bridge position, find point with minimum x
    x_values = [pt[0] for pt in centroids_map]
    min_x_index = x_values.index(min(x_values))
    x_values_nav = [pt[0] for pt in navpoints_map]
    min_x_index_nav = x_values_nav.index(min(x_values_nav))

    # Centroid coordinates (map)
    centroid_bridge = centroids_map[min_x_index]
    centroids_boxes = centroids_map[:min_x_index] + centroids_map[min_x_index + 1:]

    # Navigation point coordinates (map)
    nav_bridge = navpoints_map[min_x_index_nav]
    nav_boxes = navpoints_map[:min_x_index_nav] + navpoints_map[min_x_index_nav + 1:]
    nav_final_goals = get_centroids_map(new_img, nav_final_goals_img)

    return nav_boxes, nav_bridge, nav_final_goals