import os
import cv2
import numpy as np
import yaml


with open("./my_dataset/data.yaml", "r") as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg["names"]




def mask_to_yolo_polygon(mask, cls_id, strategy_map, simplification_factor=0.0):
    """
    Converts a binary mask to a YOLO-compatible polygon using class-specific strategy.
    Fixes misleading naming and behavior.

    Parameters:
        mask (ndarray): Binary mask.
        cls_id (int): Class index.
        simplification_factor (float): Epsilon factor for contour simplification.

    Returns:
        ndarray or None: The selected simplified polygon (Nx1x2).
    """
    
    cls_name = class_names[cls_id]
    strategy = strategy_map.get(cls_name, 0)  # default to strategy 0

    # detects boundaries of connected regions in the mask.
    # cv2.RETR_EXTERNAL gets only the outermost contours, ignoring holes inside objects.
    # cv2.CHAIN_APPROX_SIMPLE compresses points along straight lines to save memory (removes redundant points)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    if strategy == 0:
        ## Convex hull ##
        all_points = np.vstack(contours) # array of all poitns
        # Any line segment between two points inside the polygon stays entirely inside the polygon.
        # It has no "holes" or "dents"
        hull = cv2.convexHull(all_points) 
        # ArcLength is length around the polygon(contour)
        epsilon = simplification_factor * cv2.arcLength(hull, True)
        # Removes points that don’t contribute much to the shape.
        polygon = cv2.approxPolyDP(hull, epsilon, True)

    elif strategy == 1:
        ## Largest Morphological Merge ##
        # Start with a blank mask (like a clean canvas)
        merged_mask = np.zeros_like(mask, dtype=np.uint8)
        
        # Fill all the external contours into this blank mask
        # Similar to Strategy 1 where we collected all points, here we "collect" all regions as filled polygons
        cv2.fillPoly(merged_mask, contours, 1)
        
        # Apply morphological closing to merge nearby fragments and fill small gaps
        # Conceptually, this is like smoothing out irregularities before building the polygon
        merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))
        
        # Find the external contours of this cleaned-up mask
        # Like taking the convex hull points in Strategy 1, here we extract the main outer boundary
        merged_contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not merged_contours:
            return None
        
        # Select the largest contour (we focus on the main object)
        # Analogous to using all_points and convexHull in Strategy 1
        largest = max(merged_contours, key=cv2.contourArea)
        
        # Compute epsilon for simplification based on perimeter of largest contour
        # Similar to Strategy 1: arcLength gives contour perimeter, multiplied by factor for approxPolyDP
        epsilon = (simplification_factor * 0.3) * cv2.arcLength(largest, True)
        
        # Simplify the polygon by removing points that don't contribute much
        # Just like Strategy 1, this reduces complexity while keeping the overall shape
        polygon = cv2.approxPolyDP(largest, epsilon, True)

    else:
        # Fallback: largest original contour
        largest = max(contours, key=cv2.contourArea)
        epsilon = simplification_factor * cv2.arcLength(largest, True)
        polygon = cv2.approxPolyDP(largest, epsilon, True)

    return polygon
