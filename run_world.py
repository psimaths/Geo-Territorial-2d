import numpy as np
import pygame
import random
import math
from math import pi, cos, sin
from structure import (
    position_to_nearest_region,
    region_to_border_regions,
    region_to_position
)

from constants import *

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Geodesic Icosahedral Polygon Viewer")
clock = pygame.time.Clock()

# Load the world map image
world_image = pygame.image.load("world_nasa_big.png")
world_width = world_image.get_width()
world_height = world_image.get_height()



def ray_sphere_intersection(ray_origin, ray_dir, sphere_center=np.array([0, 0, 0]), sphere_radius=1.0):
    oc = ray_origin - sphere_center
    a = np.dot(ray_dir, ray_dir)
    b = 2.0 * np.dot(oc, ray_dir)
    c = np.dot(oc, oc) - sphere_radius ** 2
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    t = None
    if t1 > 0 and t2 > 0:
        t = min(t1, t2)
    elif t1 > 0:
        t = t1
    elif t2 > 0:
        t = t2
    else:
        return None
    return ray_origin + t * ray_dir

def position_to_lat_lon(position):
    """Convert 3D sphere position to latitude and longitude in radians."""
    x, y, z = position
    # Normalize the position (should already be normalized, but just in case)
    norm = np.linalg.norm(position)
    if norm > 0:
        x, y, z = x/norm, y/norm, z/norm
    
    # Calculate latitude (elevation angle from xy-plane)
    lat = math.asin(z)  # Range: [-π/2, π/2]
    
    # Calculate longitude (azimuth angle in xy-plane)
    lon = math.atan2(y, x)  # Range: [-π, π]
    
    return lat, lon

def lat_lon_to_image_coords(lat, lon, img_width, img_height):
    """Convert latitude/longitude to image pixel coordinates."""
    # Convert latitude from [-π/2, π/2] to [0, img_height-1]
    # Note: Image typically has +Y pointing down, so we flip latitude
    pixel_y = int(((-lat + pi/2) / pi) * (img_height - 1))
    
    # Convert longitude from [-π, π] to [0, img_width-1]
    pixel_x = int(((lon + pi) / (2*pi)) * (img_width - 1))
    
    # Clamp to image bounds
    pixel_x = max(0, min(img_width - 1, pixel_x))
    pixel_y = max(0, min(img_height - 1, pixel_y))
    
    return pixel_x, pixel_y

def sample_world_color(position):
    """Sample color from world image based on 3D position."""
    lat, lon = position_to_lat_lon(position)
    pixel_x, pixel_y = lat_lon_to_image_coords(lat, lon, world_width, world_height)
    
    # Get color from the world image
    color = world_image.get_at((pixel_x, pixel_y))
    return (color.r, color.g, color.b)

def assign_region_color(region_key, region_colors):
    """Assign a color to a region based on its geographic position."""
    if region_key not in region_colors:
        # Get the 3D position for this region
        position = region_to_position(region_key)
        # Sample color from world map
        region_colors[region_key] = sample_world_color(position)
    return region_colors[region_key]

# Global state



def run():
    running = True
    region_colors = {}  # Dictionary to store colors only for regions we've seen
    selected_region = None
    neighbor_regions = []

    # Global camera parameters
    cam_angle_x = 0
    cam_angle_y = pi / 4  # initial vertical angle
    cam_radius = cam_radius_default
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    ndc_x = (mx / screen_width) * 2 - 1
                    ndc_y = 1 - (my / screen_height) * 2
                    aspect = screen_width / screen_height
                    cam_pos = np.array([cam_radius * cos(cam_angle_x) * sin(cam_angle_y),
                                        cam_radius * sin(cam_angle_x) * sin(cam_angle_y),
                                        cam_radius * cos(cam_angle_y)])
                    forward = -cam_pos / np.linalg.norm(cam_pos)
                    up = np.array([0, 0, 1])
                    right = np.cross(forward, up)
                    up = np.cross(right, forward)
                    pixel_dir = forward + ndc_x * right * math.tan(fov / 2) * aspect + ndc_y * up * math.tan(fov / 2)
                    pixel_dir = pixel_dir / np.linalg.norm(pixel_dir)
                    hit = ray_sphere_intersection(cam_pos, pixel_dir)
                    if hit is not None:
                        region = position_to_nearest_region(hit)
                        selected_region = tuple(region)
                    else:
                        selected_region = None

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and selected_region is not None:
                    result = region_to_border_regions(list(selected_region))
                    neighbor_regions = [tuple(r) for r in result] if result else []
                elif event.key == pygame.K_LEFT:
                    cam_angle_x -= cam_angle_step
                elif event.key == pygame.K_RIGHT:
                    cam_angle_x += cam_angle_step
                elif event.key == pygame.K_UP:
                    cam_angle_y = max(cam_angle_y_min, cam_angle_y - cam_angle_step)
                elif event.key == pygame.K_DOWN:
                    cam_angle_y = min(pi - cam_angle_y_min, cam_angle_y + cam_angle_step)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    cam_radius = max(cam_radius_min, cam_radius - cam_radius_step)
                elif event.key == pygame.K_MINUS:
                    cam_radius += cam_radius_step

        screen.fill(color_black)
        cam_pos = np.array([cam_radius * cos(cam_angle_x) * sin(cam_angle_y),
                            cam_radius * sin(cam_angle_x) * sin(cam_angle_y),
                            cam_radius * cos(cam_angle_y)])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        up = np.cross(right, forward)
        aspect = screen_width / screen_height

        for x in range(0, screen_width, block_size):
            for y in range(0, screen_height, block_size):
                ndc_x = ((x + block_size / 2) / screen_width) * 2 - 1
                ndc_y = 1 - ((y + block_size / 2) / screen_height) * 2
                pixel_dir = forward + ndc_x * right * math.tan(fov / 2) * aspect + ndc_y * up * math.tan(fov / 2)
                pixel_dir = pixel_dir / np.linalg.norm(pixel_dir)
                hit = ray_sphere_intersection(cam_pos, pixel_dir)
                if hit is not None:
                    region = position_to_nearest_region(hit)
                    
                    region_key = tuple(region)
                    color = assign_region_color(region_key, region_colors)

                    if selected_region == region_key:
                        color = color_red
                    if selected_region is not None and region_key in neighbor_regions:
                        color = color_green

                    pygame.draw.rect(screen, color, (x, y, block_size, block_size))

        # Render FPS counter in the top right corner
        font = pygame.font.SysFont(None, 24)
        fps_text = font.render(f"FPS: {round(clock.get_fps(),3)}", True, color_white)
        screen.blit(fps_text, (screen_width - fps_text.get_width() - 10, 10))

        pygame.display.flip()
        clock.tick(target_fps)

    pygame.quit()

run()