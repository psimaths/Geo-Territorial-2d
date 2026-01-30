# Window Configuration
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "World Map Viewer"
FPS = 60

# Asset Paths
WORLD_MAP_IMAGE = "world.tif"
LAND_DATA_FILE = "bathymetry.tif"

# Zoom Configuration
ZOOM_MIN = 0.1
ZOOM_MAX = 20.0
ZOOM_SPEED = 0.1

# Pan Configuration
PAN_SPEED = 10
DRAG_SENSITIVITY = 1.0

# Colors (RGB)
BACKGROUND_COLOR = (10, 15, 30)
UI_TEXT_COLOR = (255, 255, 255)
UI_SHADOW_COLOR = (0, 0, 0)

# Land Overlay Colors (RGBA)
LAND_COLOR = (34, 139, 34, 180)
OCEAN_COLOR = (0, 105, 148, 120)

# Hexagonal Grid
HEX_WIDTH = 10  # Width of each hexagon in pixels (at 1x zoom)
HEX_BORDER_COLOR = (255, 255, 255, 200)
HEX_FILL_COLOR = (100, 200, 100, 80)
HEX_BORDER_WIDTH = 2

# UI Configuration
FONT_SIZE = 18
UI_PADDING = 15

# Game Configuration
NUM_PLAYERS = 4
STARTING_CREDITS = 1000
HEX_PRICE = 100  # Base price for purchasing a hex
EXPANSION_TIME = 10.0  # Seconds for neighbor countdown to reach zero

# Player Colors (RGBA) - distinct colors for each player
PLAYER_COLORS = [
    (220, 60, 60, 200),    # Red
    (60, 130, 220, 200),   # Blue
    (60, 180, 80, 200),    # Green
    (220, 180, 60, 200),   # Yellow
    (180, 80, 200, 200),   # Purple
    (60, 200, 200, 200),   # Cyan
    (220, 130, 60, 200),   # Orange
    (200, 100, 150, 200),  # Pink
]

# Minimum zoom level to show prices
PRICE_ZOOM_THRESHOLD = 2.0
