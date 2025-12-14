#!/usr/bin/env python3
"""
GPS Coordinate Calculator for Detected Objects
Uses equirectangular panorama projection and camera pose to estimate object GPS positions.
"""

import csv
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


# Ladybug Camera Parameters
PANORAMA_WIDTH = 16000   # pixels
PANORAMA_HEIGHT = 8000   # pixels
CAMERA_HEIGHT = 2.5      # meters (typical mounting height on vehicle)
FOV_HORIZONTAL = 360.0   # degrees
FOV_VERTICAL = 180.0     # degrees

# Default object heights for distance estimation (meters)
DEFAULT_OBJECT_HEIGHTS = {
    "stop sign": 2.2,
    "yield sign": 2.2,
    "speed limit sign": 2.5,
    "traffic light": 5.0,
    "street light pole": 6.0,
    "light pole": 6.0,
    "manhole cover": 0.0,  # ground level
    "road marking": 0.0,
    "crosswalk": 0.0,
    "pedestrian crossing": 0.0,
    "no parking sign": 2.2,
    "one way sign": 2.5,
    "road sign": 2.5,
    "parking sign": 2.2,
    "fire hydrant": 0.5,
    "traffic cone": 0.5,
    "road barrier": 0.8,
    "guardrail": 0.7,
    "curb": 0.15,
    "sidewalk": 0.0,
}


@dataclass
class CameraPose:
    """Camera position and orientation."""
    image_name: str
    timestamp: str
    latitude: float
    longitude: float
    altitude: float
    heading: float  # degrees, 0=North, clockwise
    pitch: float    # degrees
    roll: float     # degrees
    accuracy_north: float
    accuracy_east: float


@dataclass
class Detection:
    """Object detection with bounding box."""
    class_name: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class GeolocatedObject:
    """Detection with estimated GPS coordinates."""
    detection: Detection
    latitude: float
    longitude: float
    distance: float  # meters from camera
    bearing: float   # degrees from north
    elevation_angle: float  # degrees from horizontal
    confidence: str  # estimation confidence level


def load_camera_poses(csv_path: Path) -> dict[str, CameraPose]:
    """Load camera poses from CSV file."""
    poses = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 10:
                pose = CameraPose(
                    image_name=row[0] + ".jpg",
                    timestamp=row[1],
                    latitude=float(row[2]),
                    longitude=float(row[3]),
                    altitude=float(row[4]),
                    heading=float(row[5]),
                    pitch=float(row[6]),
                    roll=float(row[7]),
                    accuracy_north=float(row[8]),
                    accuracy_east=float(row[9])
                )
                poses[pose.image_name] = pose
    return poses


def pixel_to_angles(x: float, y: float) -> tuple[float, float]:
    """
    Convert pixel coordinates to spherical angles in equirectangular projection.

    Returns:
        azimuth: degrees, 0=center of image, positive=right
        elevation: degrees, 0=horizon, positive=up
    """
    # In equirectangular: x maps to longitude (azimuth), y maps to latitude (elevation)
    # Image center (8000, 4000) corresponds to azimuth=0, elevation=0

    # Azimuth: 0 at center, -180 at left edge, +180 at right edge
    azimuth = (x / PANORAMA_WIDTH) * FOV_HORIZONTAL - 180.0

    # Elevation: +90 at top, 0 at middle, -90 at bottom
    elevation = 90.0 - (y / PANORAMA_HEIGHT) * FOV_VERTICAL

    return azimuth, elevation


def estimate_distance(elevation_angle: float, object_height: float, camera_height: float = CAMERA_HEIGHT) -> Optional[float]:
    """
    Estimate distance to object using elevation angle and heights.

    For ground-level objects: distance = camera_height / tan(|elevation|)
    For elevated objects: distance = (camera_height - object_height) / tan(|elevation|)
    """
    if elevation_angle >= 0:
        # Looking up - can't estimate distance for objects above horizon without more info
        return None

    # Convert to radians
    elev_rad = math.radians(abs(elevation_angle))

    if elev_rad < 0.01:  # Nearly horizontal
        return None  # Too far to estimate reliably

    # Height difference between camera and object
    height_diff = camera_height - object_height

    if height_diff <= 0:
        return None  # Object is above camera

    # Distance on ground plane
    distance = height_diff / math.tan(elev_rad)

    return distance


def calculate_gps_offset(lat: float, lon: float, bearing: float, distance: float) -> tuple[float, float]:
    """
    Calculate new GPS position given starting point, bearing, and distance.

    Args:
        lat, lon: Starting coordinates in degrees
        bearing: Direction in degrees (0=North, clockwise)
        distance: Distance in meters

    Returns:
        new_lat, new_lon: New coordinates in degrees
    """
    # Earth radius in meters
    R = 6371000

    # Convert to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing)

    # Angular distance
    d = distance / R

    # Calculate new latitude
    new_lat_rad = math.asin(
        math.sin(lat_rad) * math.cos(d) +
        math.cos(lat_rad) * math.sin(d) * math.cos(bearing_rad)
    )

    # Calculate new longitude
    new_lon_rad = lon_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(d) * math.cos(lat_rad),
        math.cos(d) - math.sin(lat_rad) * math.sin(new_lat_rad)
    )

    return math.degrees(new_lat_rad), math.degrees(new_lon_rad)


def geolocate_detection(detection: Detection, pose: CameraPose) -> Optional[GeolocatedObject]:
    """
    Calculate GPS coordinates for a detected object.
    """
    # Get center of bounding box
    center_x = (detection.x1 + detection.x2) / 2
    center_y = (detection.y1 + detection.y2) / 2

    # Use bottom center for ground-level objects
    bottom_x = center_x
    bottom_y = detection.y2  # Bottom of bounding box

    # Convert to angles
    azimuth, elevation = pixel_to_angles(bottom_x, bottom_y)

    # Calculate absolute bearing (relative to North)
    # Camera heading is the direction the camera is facing
    # Azimuth 0 in image = camera heading direction
    absolute_bearing = (pose.heading + azimuth) % 360

    # Adjust elevation for camera pitch
    adjusted_elevation = elevation - pose.pitch

    # Get object height for distance estimation
    object_height = DEFAULT_OBJECT_HEIGHTS.get(detection.class_name, 0.0)

    # Estimate distance
    distance = estimate_distance(adjusted_elevation, object_height)

    if distance is None or distance > 100:  # Max reasonable distance
        # Can't estimate or too far
        confidence = "low"
        distance = 20.0  # Default assumption
    elif distance < 3:
        confidence = "high"
    elif distance < 15:
        confidence = "medium"
    else:
        confidence = "low"

    # Cap distance at reasonable values
    distance = min(max(distance, 1.0), 100.0)

    # Calculate GPS position
    obj_lat, obj_lon = calculate_gps_offset(
        pose.latitude, pose.longitude,
        absolute_bearing, distance
    )

    return GeolocatedObject(
        detection=detection,
        latitude=obj_lat,
        longitude=obj_lon,
        distance=distance,
        bearing=absolute_bearing,
        elevation_angle=adjusted_elevation,
        confidence=confidence
    )


def geolocate_detections(
    image_name: str,
    detections: list[dict],
    poses: dict[str, CameraPose]
) -> list[dict]:
    """
    Geolocate all detections for an image.

    Args:
        image_name: Name of the panoramic image
        detections: List of detection dictionaries with bbox, class, score
        poses: Dictionary of camera poses by image name

    Returns:
        List of geolocated detection dictionaries
    """
    if image_name not in poses:
        return []

    pose = poses[image_name]
    results = []

    for det_dict in detections:
        detection = Detection(
            class_name=det_dict["class"],
            score=det_dict["score"],
            x1=det_dict["bbox"][0],
            y1=det_dict["bbox"][1],
            x2=det_dict["bbox"][2],
            y2=det_dict["bbox"][3]
        )

        geo_obj = geolocate_detection(detection, pose)

        if geo_obj:
            results.append({
                "class": detection.class_name,
                "score": detection.score,
                "bbox": det_dict["bbox"],
                "latitude": geo_obj.latitude,
                "longitude": geo_obj.longitude,
                "distance_m": round(geo_obj.distance, 1),
                "bearing_deg": round(geo_obj.bearing, 1),
                "elevation_deg": round(geo_obj.elevation_angle, 1),
                "confidence": geo_obj.confidence,
                "camera_lat": pose.latitude,
                "camera_lon": pose.longitude,
                "camera_heading": pose.heading
            })

    return results


# Global poses cache
_poses_cache: Optional[dict[str, CameraPose]] = None

def get_poses() -> dict[str, CameraPose]:
    """Get or load camera poses (cached)."""
    global _poses_cache
    if _poses_cache is None:
        csv_path = Path("/home/giobbe/tools/data/cities/Piacenza/run1/Trigger/import_locations.csv")
        _poses_cache = load_camera_poses(csv_path)
    return _poses_cache


if __name__ == "__main__":
    # Test with sample data
    poses = get_poses()
    print(f"Loaded {len(poses)} camera poses")

    # Test with first image
    test_image = "ladybug_panoramic_4502.jpg"
    if test_image in poses:
        pose = poses[test_image]
        print(f"\nTest image: {test_image}")
        print(f"  Camera position: {pose.latitude:.6f}, {pose.longitude:.6f}")
        print(f"  Heading: {pose.heading:.1f}°")

        # Test detection (from the CSV we saw earlier)
        test_det = {
            "class": "road sign",
            "score": 0.77,
            "bbox": [15942.6, 4016.4, 16000.7, 4074.9]
        }

        results = geolocate_detections(test_image, [test_det], poses)
        if results:
            r = results[0]
            print(f"\nGeolocated detection:")
            print(f"  Class: {r['class']} ({r['score']:.2f})")
            print(f"  Object GPS: {r['latitude']:.6f}, {r['longitude']:.6f}")
            print(f"  Distance: {r['distance_m']}m, Bearing: {r['bearing_deg']}°")
            print(f"  Confidence: {r['confidence']}")
