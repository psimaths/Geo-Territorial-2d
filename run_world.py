import numpy as np
import pygame
import random
import math
from math import pi, cos, sin
from structure import (
    position_to_nearest_region,
    all_regions, region_to_border_regions,
    region_to_position
)

# Initialize pygame
pygame.init()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Geodesic Icosahedral Polygon Viewer")
clock = pygame.time.Clock()



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

def update_geodesic():
    regions = all_regions()
    colors = {}
    for region in regions:
        key = tuple(region)
        colors[key] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    return colors

# Global state



def run():
    running = True
    vertex_colors = update_geodesic()
    selected_region = None
    neighbor_regions = []
    block_size = 1

    # Global camera parameters
    cam_angle_x = 0
    cam_angle_y = pi / 4  # initial vertical angle
    cam_radius = 3.0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    ndc_x = (mx / screen_width) * 2 - 1
                    ndc_y = 1 - (my / screen_height) * 2
                    fov = pi / 3
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
                    cam_angle_x -= 0.1
                elif event.key == pygame.K_RIGHT:
                    cam_angle_x += 0.1
                elif event.key == pygame.K_UP:
                    cam_angle_y = max(0.1, cam_angle_y - 0.1)
                elif event.key == pygame.K_DOWN:
                    cam_angle_y = min(pi - 0.1, cam_angle_y + 0.1)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    cam_radius = max(1.5, cam_radius - 0.2)
                elif event.key == pygame.K_MINUS:
                    cam_radius += 0.2

        screen.fill((0, 0, 0))
        cam_pos = np.array([cam_radius * cos(cam_angle_x) * sin(cam_angle_y),
                            cam_radius * sin(cam_angle_x) * sin(cam_angle_y),
                            cam_radius * cos(cam_angle_y)])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        up = np.cross(right, forward)
        fov = pi / 3
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
                    color = vertex_colors.get(region_key, (255, 255, 255))

                    if selected_region == region_key:
                        color = (255, 0, 0)
                    if selected_region is not None and region_key in neighbor_regions:
                        color = (0, 255, 0)

                    pygame.draw.rect(screen, color, (x, y, block_size, block_size))

        # Render FPS counter in the top right corner
        font = pygame.font.SysFont(None, 24)
        fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, (255, 255, 255))
        screen.blit(fps_text, (screen_width - fps_text.get_width() - 10, 10))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

run()