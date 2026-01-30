import pygame
import sys
import os
import math
import time
import numpy as np
from PIL import Image
import io

# Allow large images (world.tif is ~233 million pixels)
Image.MAX_IMAGE_PIXELS = 300000000

from constants import *


class Player:
    """Represents a player in the game."""
    
    def __init__(self, player_id, color):
        self.id = player_id
        self.color = color
        self.credits = STARTING_CREDITS
        self.owned_hexes = set()  # Set of (col, row) tuples
    
    def can_afford(self, price):
        return self.credits >= price
    
    def spend(self, amount):
        self.credits -= amount
    
    def add_hex(self, col, row):
        self.owned_hexes.add((col, row))


class HexState:
    """Tracks ownership and countdown state for a hex."""
    
    def __init__(self):
        self.owner = None  # Player ID or None
        # Multiple players can have countdowns running simultaneously
        # Maps player_id -> timer_start_time
        self.pending_timers = {}
        self.timer_duration = EXPANSION_TIME
    
    def is_owned(self):
        return self.owner is not None
    
    def is_pending(self):
        return len(self.pending_timers) > 0 and not self.is_owned()
    
    def is_pending_for(self, player_id):
        """Check if a specific player has a countdown running."""
        return player_id in self.pending_timers and not self.is_owned()
    
    def get_progress(self, player_id=None):
        """Returns countdown progress from 0.0 to 1.0 for a specific player."""
        if player_id is None:
            # Return the fastest progress (for determining who wins)
            if not self.pending_timers:
                return 0.0
            return max(self.get_progress(pid) for pid in self.pending_timers)
        
        if player_id not in self.pending_timers or self.is_owned():
            return 0.0
        elapsed = time.time() - self.pending_timers[player_id]
        return min(1.0, elapsed / self.timer_duration)
    
    def get_current_price(self, player_id=None):
        """Returns the current price based on countdown progress for a player."""
        if player_id is not None and player_id in self.pending_timers:
            progress = self.get_progress(player_id)
            return int(HEX_PRICE * (1.0 - progress))
        return HEX_PRICE
    
    def get_leading_player(self):
        """Returns the player_id with the most progress (closest to claiming)."""
        if not self.pending_timers:
            return None
        return max(self.pending_timers.keys(), key=lambda pid: self.get_progress(pid))
    
    def start_countdown(self, player_id):
        """Start countdown for a player to claim this hex."""
        # Only start if this player doesn't already have a countdown here
        if player_id not in self.pending_timers:
            self.pending_timers[player_id] = time.time()
    
    def claim(self, player_id):
        """Immediately claim this hex for a player."""
        self.owner = player_id
        self.pending_timers.clear()  # Clear all pending countdowns
    
    def remove_pending(self, player_id):
        """Remove a specific player's pending countdown."""
        if player_id in self.pending_timers:
            del self.pending_timers[player_id]


class LandProcessor:
    """Processes bathymetry TIF to create land/ocean mask."""
    
    def __init__(self, tif_path, target_width, target_height):
        self.tif_path = tif_path
        self.target_width = target_width
        self.target_height = target_height
        self.land_mask = None
        self.land_overlay = None
        
    def load_and_process(self):
        """Load the TIF and create land mask."""
        print(f"Loading {self.tif_path}...")
        
        img = Image.open(self.tif_path)
        print(f"Original size: {img.size}")
        
        if img.size != (self.target_width, self.target_height):
            print(f"Resizing to {self.target_width}x{self.target_height}")
            img = img.resize((self.target_width, self.target_height), Image.Resampling.NEAREST)
        
        img_array = np.array(img)
        
        # Handle both grayscale and RGB
        if len(img_array.shape) == 2:
            brightness = img_array.astype(np.float32)
        else:
            brightness = np.mean(img_array[:, :, :3], axis=2).astype(np.float32)
        
        max_val = brightness.max()
        
        self.land_mask = brightness == max_val
        
        land_pct = np.sum(self.land_mask) / self.land_mask.size * 100
        print(f"Land coverage: {land_pct:.1f}%")
        
        self._create_overlay()
        return True
    
    def _create_overlay(self):
        """Create RGBA overlay for rendering."""
        h, w = self.land_mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
        overlay[self.land_mask, :] = LAND_COLOR
        overlay[~self.land_mask, :] = OCEAN_COLOR
        
        pil_img = Image.fromarray(overlay, 'RGBA')
        
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)
        
        self.land_overlay = pygame.image.load(buffer).convert_alpha()
        print(f"Overlay created: {self.land_overlay.get_size()}")
    
    def is_land(self, x, y):
        """Check if a point is on land."""
        if self.land_mask is None:
            return False
        
        # Clamp to valid range
        ix = max(0, min(int(x), self.target_width - 1))
        iy = max(0, min(int(y), self.target_height - 1))
        
        return self.land_mask[iy, ix]


class HexGrid:
    """Hexagonal grid overlay that only renders on land."""
    
    def __init__(self, hex_width, map_width, map_height, land_processor):
        self.hex_width = hex_width
        self.map_width = map_width
        self.map_height = map_height
        self.land_processor = land_processor
        
        # Hex geometry (pointy-top hexagons)
        self.hex_height = hex_width * math.sqrt(3) / 2
        self.row_height = self.hex_height * 0.75
        
        # Calculate grid dimensions
        self.cols = int(map_width / (hex_width * 0.75)) + 2
        self.rows = int(map_height / self.hex_height) + 2
        
        # Pre-calculate which hexes are on land
        self.land_hexes = self._calculate_land_hexes()
        self.land_hexes_set = set(self.land_hexes)  # For fast lookup
        print(f"Hex grid: {self.cols}x{self.rows}, {len(self.land_hexes)} land hexes")
    
    def _hex_center(self, col, row):
        """Get the center of a hex in map coordinates."""
        x = col * self.hex_width * 0.75
        y = row * self.hex_height
        
        # Offset odd columns
        if col % 2 == 1:
            y += self.hex_height / 2
        
        return x, y
    
    def _hex_corners(self, cx, cy, size):
        """Get the 6 corners of a pointy-top hexagon."""
        corners = []
        for i in range(6):
            angle = i * math.pi / 3
            x = cx + size * math.cos(angle)
            y = cy + size * math.sin(angle)
            corners.append((x, y))
        return corners
    
    def _calculate_land_hexes(self):
        """Determine which hexes are on land."""
        land_hexes = []
        
        for col in range(self.cols):
            for row in range(self.rows):
                cx, cy = self._hex_center(col, row)
                
                # Check if center is on land
                if self.land_processor and self.land_processor.is_land(cx, cy):
                    land_hexes.append((col, row))
        
        return land_hexes
    
    def get_neighbors(self, col, row):
        """Get the 6 neighboring hex coordinates for pointy-top hex grid."""
        if col % 2 == 0:  # Even column
            neighbors = [
                (col - 1, row - 1), (col - 1, row),
                (col, row - 1), (col, row + 1),
                (col + 1, row - 1), (col + 1, row)
            ]
        else:  # Odd column
            neighbors = [
                (col - 1, row), (col - 1, row + 1),
                (col, row - 1), (col, row + 1),
                (col + 1, row), (col + 1, row + 1)
            ]
        # Filter to valid land hexes
        return [(c, r) for c, r in neighbors if (c, r) in self.land_hexes_set]
    
    def screen_to_hex(self, screen_x, screen_y, offset_x, offset_y, zoom):
        """Convert screen coordinates to hex grid coordinates."""
        # Convert screen coords to map coords
        map_x = (screen_x - offset_x) / zoom
        map_y = (screen_y - offset_y) / zoom
        
        # Find closest hex
        best_hex = None
        best_dist = float('inf')
        
        for col, row in self.land_hexes:
            cx, cy = self._hex_center(col, row)
            dist = (cx - map_x) ** 2 + (cy - map_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_hex = (col, row)
        
        # Check if click is within hex bounds (approximate)
        if best_hex:
            cx, cy = self._hex_center(best_hex[0], best_hex[1])
            max_dist = (self.hex_width / 2) ** 2
            if best_dist <= max_dist:
                return best_hex
        
        return None
    
    def render(self, screen, offset_x, offset_y, zoom, hex_states=None, players=None, current_player_id=None, font=None):
        """Render visible land hexes with ownership and countdown indicators."""
        screen_w, screen_h = screen.get_size()
        
        hex_size = (self.hex_width / 2) * zoom
        
        # Skip if hexes would be too small to see
        if hex_size < 3:
            return
        
        for col, row in self.land_hexes:
            cx, cy = self._hex_center(col, row)
            
            # Transform to screen coordinates
            screen_cx = offset_x + cx * zoom
            screen_cy = offset_y + cy * zoom
            
            # Skip if off screen (with margin)
            margin = hex_size * 2
            if (screen_cx < -margin or screen_cx > screen_w + margin or
                screen_cy < -margin or screen_cy > screen_h + margin):
                continue
            
            # Get corners in screen space
            corners = self._hex_corners(screen_cx, screen_cy, hex_size)
            
            # Determine fill color based on ownership
            hex_state = hex_states.get((col, row)) if hex_states else None
            fill_color = HEX_FILL_COLOR
            
            if hex_state and hex_state.is_owned() and players:
                # Use owner's color
                owner = players[hex_state.owner]
                fill_color = owner.color
            
            # Draw filled hex
            if len(fill_color) == 4 and fill_color[3] < 255:
                hex_surface = pygame.Surface((int(hex_size * 2.5), int(hex_size * 2.5)), pygame.SRCALPHA)
                local_corners = [(x - screen_cx + hex_size * 1.25, y - screen_cy + hex_size * 1.25) for x, y in corners]
                pygame.draw.polygon(hex_surface, fill_color, local_corners)
                screen.blit(hex_surface, (screen_cx - hex_size * 1.25, screen_cy - hex_size * 1.25))
            else:
                pygame.draw.polygon(screen, fill_color[:3], corners)
            
            # Draw radial progress indicator for pending hexes (current player only)
            if hex_state and hex_state.is_pending_for(current_player_id):
                progress = hex_state.get_progress(current_player_id)
                if progress > 0:
                    self._draw_radial_progress(screen, screen_cx, screen_cy, hex_size, progress, players[current_player_id].color)
            
            # Draw border
            pygame.draw.polygon(screen, HEX_BORDER_COLOR[:3], corners, HEX_BORDER_WIDTH)
            
            # Draw price when zoomed in (for current player's pending or unowned hexes)
            if zoom >= PRICE_ZOOM_THRESHOLD and font and hex_state:
                if not hex_state.is_owned():
                    # Show price for unowned hexes (current player's discounted price if they have a countdown)
                    price = hex_state.get_current_price(current_player_id)
                    self._draw_price(screen, screen_cx, screen_cy, price, font)
    
    def _draw_radial_progress(self, screen, cx, cy, hex_size, progress, color):
        """Draw a radial progress indicator (pie slice) on a hex."""
        # Create surface for the arc
        surface_size = int(hex_size * 2.5)
        surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
        center = (surface_size // 2, surface_size // 2)
        radius = int(hex_size * 0.8)
        
        # Calculate arc angles (start from top, go clockwise)
        # pygame.draw.arc uses radians and counter-clockwise, so we adjust
        start_angle = math.pi / 2  # Start from top
        end_angle = start_angle - (progress * 2 * math.pi)  # Go clockwise
        
        # Draw filled pie slice using polygon points
        num_points = max(3, int(progress * 30))  # More points for smoother arc
        points = [center]
        for i in range(num_points + 1):
            angle = start_angle - (i / num_points) * progress * 2 * math.pi
            x = center[0] + radius * math.cos(angle)
            y = center[1] - radius * math.sin(angle)  # Negative because y increases downward
            points.append((x, y))
        
        if len(points) >= 3:
            # Use a semi-transparent version of the player's color
            pie_color = (color[0], color[1], color[2], 150)
            pygame.draw.polygon(surface, pie_color, points)
        
        screen.blit(surface, (cx - surface_size // 2, cy - surface_size // 2))
    
    def _draw_price(self, screen, cx, cy, price, font):
        """Draw the hex price at the center."""
        text = f"${price}"
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(int(cx), int(cy)))
        
        # Draw shadow
        shadow_surface = font.render(text, True, (0, 0, 0))
        shadow_rect = shadow_surface.get_rect(center=(int(cx) + 1, int(cy) + 1))
        screen.blit(shadow_surface, shadow_rect)
        screen.blit(text_surface, text_rect)


class WorldMapViewer:
    def __init__(self):
        pygame.init()
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        
        self.load_map()
        self.load_land_data()
        self.create_hex_grid()
        self.init_game_state()
        
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        self.show_land_overlay = False  # Off by default
        self.show_hex_grid = True
        
        self.dragging = False
        self.drag_start = (0, 0)
        self.offset_start = (0, 0)
        
        self.font = pygame.font.SysFont("Menlo, Monaco, monospace", FONT_SIZE)
        self.small_font = pygame.font.SysFont("Menlo, Monaco, monospace", 12)
        
        self.center_map()
    
    def init_game_state(self):
        """Initialize players and hex states."""
        # Create players
        self.players = []
        for i in range(NUM_PLAYERS):
            color = PLAYER_COLORS[i % len(PLAYER_COLORS)]
            self.players.append(Player(i, color))
        
        self.current_player_index = 0
        
        # Initialize hex states for all land hexes
        self.hex_states = {}
        if self.hex_grid:
            for col, row in self.hex_grid.land_hexes:
                self.hex_states[(col, row)] = HexState()
    
    def get_current_player(self):
        """Get the currently active player."""
        return self.players[self.current_player_index]
    
    def cycle_player(self, direction):
        """Cycle to next or previous player."""
        self.current_player_index = (self.current_player_index + direction) % len(self.players)
    
    def try_purchase_hex(self, screen_x, screen_y):
        """Attempt to purchase a hex at the given screen coordinates."""
        if not self.hex_grid:
            return False
        
        # Convert screen coords to hex
        hex_coords = self.hex_grid.screen_to_hex(screen_x, screen_y, self.offset_x, self.offset_y, self.zoom)
        if hex_coords is None:
            return False
        
        col, row = hex_coords
        
        # Check if this is a land hex
        if (col, row) not in self.hex_states:
            return False
        
        hex_state = self.hex_states[(col, row)]
        
        # Check if already owned
        if hex_state.is_owned():
            return False
        
        player = self.get_current_player()
        # Get the price specific to this player (discounted if they have a countdown)
        price = hex_state.get_current_price(player.id)
        
        # Check if player can afford it
        if not player.can_afford(price):
            return False
        
        # Purchase the hex
        player.spend(price)
        hex_state.claim(player.id)
        player.add_hex(col, row)
        
        # Start countdown on neighboring unowned land hexes
        self.start_neighbor_countdowns(col, row, player.id)
        
        return True
    
    def start_neighbor_countdowns(self, col, row, player_id):
        """Start countdown timers on all unowned neighboring hexes."""
        if not self.hex_grid:
            return
        
        neighbors = self.hex_grid.get_neighbors(col, row)
        for ncol, nrow in neighbors:
            if (ncol, nrow) in self.hex_states:
                hex_state = self.hex_states[(ncol, nrow)]
                # Only start countdown if not owned and not already pending for this player
                if not hex_state.is_owned() and not hex_state.is_pending_for(player_id):
                    hex_state.start_countdown(player_id)
    
    def update_countdowns(self):
        """Check all pending hexes and claim any that have finished counting down."""
        hexes_to_claim = []
        
        # First pass: find all hexes where any player's countdown has completed
        for coords, hex_state in self.hex_states.items():
            if hex_state.is_pending():
                # Check each player's countdown
                for player_id in list(hex_state.pending_timers.keys()):
                    if hex_state.get_progress(player_id) >= 1.0:
                        # This player's countdown finished first
                        hexes_to_claim.append((coords, player_id))
                        break  # Only one player can claim
        
        # Second pass: claim them and start new countdowns
        for (col, row), player_id in hexes_to_claim:
            hex_state = self.hex_states[(col, row)]
            hex_state.claim(player_id)
            
            # Add to player's owned hexes
            self.players[player_id].add_hex(col, row)
            
            # Start countdown on neighbors
            self.start_neighbor_countdowns(col, row, player_id)
    
    def load_map(self):
        """Load the world map image."""
        map_path = os.path.join(os.path.dirname(__file__), WORLD_MAP_IMAGE)
        
        if not os.path.exists(map_path):
            print(f"Error: Could not find {WORLD_MAP_IMAGE}")
            sys.exit(1)
        
        print(f"Loading {WORLD_MAP_IMAGE}...")
        
        if map_path.lower().endswith(('.tif', '.tiff')):
            img = Image.open(map_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            self.original_image = pygame.image.load(buffer).convert()
        else:
            self.original_image = pygame.image.load(map_path).convert()
        
        self.image_width = self.original_image.get_width()
        self.image_height = self.original_image.get_height()
        print(f"Base map: {self.image_width}x{self.image_height}")
    
    def load_land_data(self):
        """Load and process the land data file."""
        land_path = os.path.join(os.path.dirname(__file__), LAND_DATA_FILE)
        
        if not os.path.exists(land_path):
            print(f"Warning: {LAND_DATA_FILE} not found")
            self.land_processor = None
            return
        
        self.land_processor = LandProcessor(land_path, self.image_width, self.image_height)
        if not self.land_processor.load_and_process():
            self.land_processor = None
    
    def create_hex_grid(self):
        """Create the hexagonal grid."""
        if self.land_processor:
            self.hex_grid = HexGrid(HEX_WIDTH, self.image_width, self.image_height, self.land_processor)
        else:
            self.hex_grid = None
    
    def center_map(self):
        screen_w, screen_h = self.screen.get_size()
        scaled_w = self.image_width * self.zoom
        scaled_h = self.image_height * self.zoom
        self.offset_x = (screen_w - scaled_w) / 2
        self.offset_y = (screen_h - scaled_h) / 2
        self.clamp_offset()
    
    def clamp_offset(self):
        screen_w, screen_h = self.screen.get_size()
        scaled_w = self.image_width * self.zoom
        scaled_h = self.image_height * self.zoom
        
        if scaled_w <= screen_w:
            self.offset_x = (screen_w - scaled_w) / 2
        else:
            self.offset_x = min(0, max(screen_w - scaled_w, self.offset_x))
        
        if scaled_h <= screen_h:
            self.offset_y = (screen_h - scaled_h) / 2
        else:
            self.offset_y = min(0, max(screen_h - scaled_h, self.offset_y))
    
    def render_visible_portion(self, source_image):
        """Render only the visible portion of an image."""
        screen_w, screen_h = self.screen.get_size()
        
        img_w = source_image.get_width()
        img_h = source_image.get_height()
        
        scale_x = img_w / self.image_width
        scale_y = img_h / self.image_height
        
        vis_left = -self.offset_x
        vis_top = -self.offset_y
        vis_right = vis_left + screen_w
        vis_bottom = vis_top + screen_h
        
        src_left = max(0, (vis_left / self.zoom) * scale_x)
        src_top = max(0, (vis_top / self.zoom) * scale_y)
        src_right = min(img_w, (vis_right / self.zoom) * scale_x)
        src_bottom = min(img_h, (vis_bottom / self.zoom) * scale_y)
        
        src_x = int(src_left)
        src_y = int(src_top)
        src_w = int(src_right - src_left)
        src_h = int(src_bottom - src_top)
        
        if src_w <= 0 or src_h <= 0:
            return
        
        dst_w = int((src_w / scale_x) * self.zoom)
        dst_h = int((src_h / scale_y) * self.zoom)
        
        if dst_w <= 0 or dst_h <= 0:
            return
        
        max_dim = 4096
        if dst_w > max_dim or dst_h > max_dim:
            sf = min(max_dim / dst_w, max_dim / dst_h)
            dst_w = int(dst_w * sf)
            dst_h = int(dst_h * sf)
        
        src_rect = pygame.Rect(src_x, src_y, src_w, src_h)
        try:
            visible = source_image.subsurface(src_rect)
        except ValueError:
            return
        
        try:
            scaled = pygame.transform.smoothscale(visible, (dst_w, dst_h))
        except pygame.error:
            scaled = pygame.transform.scale(visible, (dst_w, dst_h))
        
        dst_x = int(self.offset_x + (src_x / scale_x) * self.zoom)
        dst_y = int(self.offset_y + (src_y / scale_y) * self.zoom)
        
        self.screen.blit(scaled, (dst_x, dst_y))
    
    def handle_zoom(self, zoom_in, mouse_pos):
        old_zoom = self.zoom
        
        if zoom_in:
            self.zoom = min(ZOOM_MAX, self.zoom * (1 + ZOOM_SPEED))
        else:
            self.zoom = max(ZOOM_MIN, self.zoom * (1 - ZOOM_SPEED))
        
        factor = self.zoom / old_zoom
        mx, my = mouse_pos
        
        self.offset_x = mx - (mx - self.offset_x) * factor
        self.offset_y = my - (my - self.offset_y) * factor
        
        self.clamp_offset()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    self.zoom = 1.0
                    self.center_map()
                elif event.key == pygame.K_l:
                    if self.land_processor:
                        self.show_land_overlay = not self.show_land_overlay
                elif event.key == pygame.K_h:
                    self.show_hex_grid = not self.show_hex_grid
                elif event.key == pygame.K_PLUS:
                    self.handle_zoom(True, self.screen.get_rect().center)
                # Player cycling: - for previous, = for next
                elif event.key == pygame.K_MINUS:
                    self.cycle_player(-1)
                elif event.key == pygame.K_EQUALS:
                    self.cycle_player(1)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.dragging = False
                    self.drag_start = event.pos
                    self.offset_start = (self.offset_x, self.offset_y)
                    self.click_start = event.pos
                elif event.button == 4:
                    self.handle_zoom(True, event.pos)
                elif event.button == 5:
                    self.handle_zoom(False, event.pos)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    # If we didn't drag much, treat as a click to purchase
                    if hasattr(self, 'click_start') and self.click_start:
                        dx = abs(event.pos[0] - self.click_start[0])
                        dy = abs(event.pos[1] - self.click_start[1])
                        if dx < 5 and dy < 5:
                            # This was a click, try to purchase hex
                            self.try_purchase_hex(event.pos[0], event.pos[1])
                    self.dragging = False
                    self.click_start = None
            
            elif event.type == pygame.MOUSEMOTION:
                # Start dragging if mouse moved enough after mousedown
                if hasattr(self, 'click_start') and self.click_start and not self.dragging:
                    dx = abs(event.pos[0] - self.click_start[0])
                    dy = abs(event.pos[1] - self.click_start[1])
                    if dx >= 5 or dy >= 5:
                        self.dragging = True
                
                if self.dragging:
                    dx = (event.pos[0] - self.drag_start[0]) * DRAG_SENSITIVITY
                    dy = (event.pos[1] - self.drag_start[1]) * DRAG_SENSITIVITY
                    self.offset_x = self.offset_start[0] + dx
                    self.offset_y = self.offset_start[1] + dy
                    self.clamp_offset()
            
            elif event.type == pygame.MOUSEWHEEL:
                pos = pygame.mouse.get_pos()
                self.handle_zoom(event.y > 0, pos)
            
            elif event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.clamp_offset()
        
        keys = pygame.key.get_pressed()
        panned = False
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.offset_x += PAN_SPEED
            panned = True
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.offset_x -= PAN_SPEED
            panned = True
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.offset_y += PAN_SPEED
            panned = True
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.offset_y -= PAN_SPEED
            panned = True
        
        if panned:
            self.clamp_offset()
        
        return True
    
    def draw_text(self, text, x, y):
        shadow = self.font.render(text, True, UI_SHADOW_COLOR)
        label = self.font.render(text, True, UI_TEXT_COLOR)
        self.screen.blit(shadow, (x + 1, y + 1))
        self.screen.blit(label, (x, y))
    
    def render(self):
        self.screen.fill(BACKGROUND_COLOR)
        
        self.render_visible_portion(self.original_image)
        
        if self.show_land_overlay and self.land_processor and self.land_processor.land_overlay:
            self.render_visible_portion(self.land_processor.land_overlay)
        
        if self.show_hex_grid and self.hex_grid:
            self.hex_grid.render(
                self.screen, self.offset_x, self.offset_y, self.zoom,
                hex_states=self.hex_states,
                players=self.players,
                current_player_id=self.current_player_index,
                font=self.small_font
            )
        
        y = UI_PADDING
        
        # Draw current player info with color swatch
        player = self.get_current_player()
        player_text = f"Player {player.id + 1}"
        self.draw_text(player_text, UI_PADDING, y)
        
        # Draw color swatch next to player name
        swatch_x = UI_PADDING + self.font.size(player_text)[0] + 10
        swatch_rect = pygame.Rect(swatch_x, y + 2, 20, FONT_SIZE - 4)
        pygame.draw.rect(self.screen, player.color[:3], swatch_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), swatch_rect, 1)
        y += FONT_SIZE + 5
        
        # Draw credits
        self.draw_text(f"Credits: ${player.credits}", UI_PADDING, y)
        y += FONT_SIZE + 5
        
        # Draw owned hex count
        self.draw_text(f"Hexes: {len(player.owned_hexes)}", UI_PADDING, y)
        y += FONT_SIZE + 10
        
        # Draw zoom
        self.draw_text(f"Zoom: {self.zoom:.2f}x", UI_PADDING, y)
        y += FONT_SIZE + 5
        
        hex_status = "ON" if self.show_hex_grid else "OFF"
        self.draw_text(f"Hex Grid: {hex_status} (H)", UI_PADDING, y)
        y += FONT_SIZE + 5
        
        overlay_status = "ON" if self.show_land_overlay else "OFF"
        self.draw_text(f"Land Overlay: {overlay_status} (L)", UI_PADDING, y)
        
        help_y = self.screen.get_height() - UI_PADDING - FONT_SIZE
        self.draw_text("Click: Buy | -/=: Switch Player | Scroll: Zoom | Drag/WASD: Pan | H: Hex | R: Reset", UI_PADDING, help_y)
        
        pygame.display.flip()
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update_countdowns()
            self.render()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    viewer = WorldMapViewer()
    viewer.run()
