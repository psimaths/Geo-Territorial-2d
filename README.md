# Geoterritorial

A geodesic icosahedral visualization system that allows interaction with a subdivided spherical structure.

## Overview

Geoterritorial is a Python-based application that visualizes and enables interaction with a geodesic icosahedral structure (a sphere divided into polygonal regions). The project represents the Earth's surface as a geodesic dome, dividing the sphere into regions that can be selected and analyzed.

## Features

- **3D Geodesic Visualization**: Renders a 3D icosahedral sphere model onto a 2D screen
- **Interactive Navigation**: Rotate the view, zoom in/out, and select specific regions
- **Region Selection**: Click on any region to highlight it
- **Neighbor Analysis**: View all adjacent regions to a selected region
- **Real-time Rendering**: Efficient ray-sphere intersection algorithm for visualization

## Project Structure

The project consists of two main Python files:

### structure.py

Contains the core data structures and mathematical functions for representing the geodesic structure:

- Definition of icosahedron vertices and faces
- Functions for region management (corners, edges, faces)
- Mathematical transformations between 3D positions and region identifiers
- Neighbor region calculations
- Optimized spatial lookup functions

### run_world.py

Handles the visualization and interaction aspects:

- Pygame-based rendering engine
- User input handling
- Camera controls and view manipulation
- Ray-casting for region selection
- Visual feedback system (region highlighting, neighboring regions)

## How It Works

The application uses a mathematical model of an icosahedron (20-faced polyhedron) and subdivides each face into smaller triangular regions. These regions are identified by their position (corner, edge, or face) and can be selected and analyzed.

The `edge_length` parameter (default: 30) determines the resolution of the subdivision.

## Controls

- **Arrow Keys**: Rotate the view
- **+/-**: Zoom in/out
- **Left Mouse Click**: Select a region
- **Space**: Show neighboring regions of the selected region (highlighted in green)
- **ESC/Close Window**: Exit the application

## Requirements

- Python 3.x
- NumPy
- PyGame

## Getting Started

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install numpy pygame
   ```
3. Run the application:
   ```
   python run_world.py
   ```

## Technical Details

The project uses several advanced techniques:

- **Ray-Sphere Intersection**: For accurately projecting 2D screen coordinates onto the 3D sphere
- **Vector Mathematics**: Optimized with NumPy for efficient operations
- **Precomputed Transformations**: Matrix operations are precomputed for performance
- **Cached Lookups**: Spatial relationships are cached in dictionaries for O(1) access