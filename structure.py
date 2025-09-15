import numpy as np
import itertools

# =============================================================================
# Global Constants and Precomputed Data
# =============================================================================

edge_length = 100

# Convert corner positions to a NumPy array for vectorized operations.
corner_positions = np.array([
    [0.0, 0.0, 1.0],
    [0.8944271909999159, 0.0, 0.4472135954999579],
    [0.27639320225002106, 0.8506508083520399, 0.4472135954999579],
    [-0.7236067977499788, 0.5257311121191337, 0.4472135954999579],
    [-0.723606797749979, -0.5257311121191335, 0.4472135954999579],
    [0.27639320225002084, -0.85065080835204, 0.4472135954999579],
    [0.7236067977499789, 0.5257311121191336, -0.4472135954999579],
    [-0.27639320225002095, 0.85065080835204, -0.4472135954999579],
    [-0.8944271909999159, 1.0953573965284052e-16, -0.4472135954999579],
    [-0.2763932022500211, -0.8506508083520399, -0.4472135954999579],
    [0.7236067977499788, -0.5257311121191338, -0.4472135954999579],
    [0.0, 0.0, -1.0]
])

edge_number_to_vertices = [
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
    [1, 2], [2, 3], [3, 4], [4, 5], [1, 5],
    [1, 10], [1, 6], [2, 6], [2, 7], [3, 7], [3, 8], [4, 8], [4, 9], [5, 9], [5, 10],
    [6, 7], [7, 8], [8, 9], [9, 10], [6, 10],
    [6, 11], [7, 11], [8, 11], [9, 11], [10, 11]
]

face_number_to_corners = [
    [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 1, 5],
    [1, 2, 6], [2, 3, 7], [3, 4, 8], [4, 5, 9], [1, 5, 10],
    [1, 6, 10], [2, 6, 7], [3, 7, 8], [4, 8, 9], [5, 9, 10],
    [6, 7, 11], [7, 8, 11], [8, 9, 11], [9, 10, 11], [6, 10, 11]
]


# -------------------------------------------------------------------------
# Build dictionaries for O(1) lookups.
# -------------------------------------------------------------------------
# Map an edge (sorted tuple of vertex indices) to its edge number.
edge_lookup = {tuple(sorted(edge)): i for i, edge in enumerate(edge_number_to_vertices)}

# Map sorted corner triplets (as a tuple) to face number.
face_lookup = {tuple(face): i for i, face in enumerate(face_number_to_corners)}

# Map each edge (as a sorted tuple) to a list of (face, other_corner) that share that edge.
edge_to_faces = {}
for face, corners in enumerate(face_number_to_corners):
    for pair in itertools.combinations(corners, 2):
        key = tuple(sorted(pair))
        other = (set(corners) - set(key)).pop()
        edge_to_faces.setdefault(key, []).append((face, other))

# -------------------------------------------------------------------------
# Precompute transformation matrices for each face.
# For each face we compute:
#   - face_transform[face]: a 3x3 matrix whose rows are:
#         v1 = (c1-c0)/edge_length,
#         v2 = (c2-c0)/edge_length,
#         v3 = np.cross(c1-c0, c2-c0)
#   - face_inv_transform[face]: the inverse of the above matrix.
#   - face_origin[face]: the corner position for c0.
# -------------------------------------------------------------------------
face_transform = {}
face_inv_transform = {}
face_origin = {}
for face, corners in enumerate(face_number_to_corners):
    c0 = corner_positions[corners[0]]
    c1 = corner_positions[corners[1]]
    c2 = corner_positions[corners[2]]
    v1 = (c1 - c0) / edge_length
    v2 = (c2 - c0) / edge_length
    v3 = np.cross(c1 - c0, c2 - c0)
    matrix = np.array([v1, v2, v3])
    face_transform[face] = matrix
    face_origin[face] = c0
    face_inv_transform[face] = np.linalg.inv(matrix)


# =============================================================================
# Utility Functions
# =============================================================================

def all_regions_on_face(face_num: int) -> list:
    regions = []
    for c0_c1 in range(1, edge_length):
        for c0_c2 in range(1, edge_length - c0_c1):
            regions.append((face_num, c0_c1, c0_c2))
    return regions


def all_regions_on_edge(edge_num: int) -> list:
    return [(edge_num, pos) for pos in range(1, edge_length)]


def all_face_regions() -> list:
    regions = []
    for face in range(len(face_number_to_corners)):
        regions.extend(all_regions_on_face(face))
    return regions


def all_edge_regions() -> list:
    regions = []
    for edge in range(len(edge_number_to_vertices)):
        regions.extend(all_regions_on_edge(edge))
    return regions


def all_corner_regions() -> list:
    return [tuple([i]) for i in range(len(corner_positions))]


def all_regions() -> list:
    return all_corner_regions() + all_edge_regions() + all_face_regions()


# =============================================================================
# Core Functions (Optimized)
# =============================================================================

def region_to_border_regions(region: list) -> list:
    border_regions = []
    if len(region) == 1:
        # Corner region: for each edge attached to this corner.
        corner = region[0]
        for edge, verts in enumerate(edge_number_to_vertices):
            if corner == verts[0]:
                border_regions.append([edge, 1])
            elif corner == verts[1]:
                border_regions.append([edge, edge_length - 1])
    elif len(region) == 2:
        edge_num, pos = region
        c0, c1 = edge_number_to_vertices[edge_num]
        border_corner = None
        if pos == 1:
            border_regions.append([c0])
            border_corner = c0
        else:
            border_regions.append([edge_num, pos - 1])
        if pos == edge_length - 1:
            border_regions.append([c1])
            border_corner = c1
        else:
            border_regions.append([edge_num, pos + 1])
        # Retrieve adjacent faces using precomputed edge_to_faces.
        key = tuple(sorted((c0, c1)))
        border_faces = []
        for face, other_corner in edge_to_faces.get(key, []):
            if border_corner is not None:
                key2 = tuple(sorted((border_corner, other_corner)))
                border_edge_num = edge_lookup[key2]
                if border_corner < other_corner:
                    border_regions.append([border_edge_num, 1])
                else:
                    border_regions.append([border_edge_num, edge_length - 1])
            border_faces.append((face, other_corner))
        for face, other_corner in border_faces:
            if c1 < other_corner:
                if pos != 1:
                    border_regions.append([face, pos - 1, 1])
                if pos != edge_length - 1:
                    border_regions.append([face, pos, 1])
            elif other_corner < c0:
                if pos != 1:
                    border_regions.append([face, edge_length - pos, pos - 1])
                if pos != edge_length - 1:
                    border_regions.append([face, edge_length - pos - 1, pos])
            else:
                if pos != 1:
                    border_regions.append([face, 1, pos - 1])
                if pos != edge_length - 1:
                    border_regions.append([face, 1, pos])
    elif len(region) == 3:
        face, d1, d2 = region
        c0, c1, c2 = face_number_to_corners[face]
        # Border for edge between c0 and c1.
        if d2 == 1:
            key = tuple(sorted((c0, c1)))
            edge_num = edge_lookup[key]
            border_regions.append([edge_num, d1])
            border_regions.append([edge_num, d1 + 1])
        else:
            border_regions.append([face, d1, d2 - 1])
            border_regions.append([face, d1 + 1, d2 - 1])
        # Border for edge between c0 and c2.
        if d1 == 1:
            key = tuple(sorted((c0, c2)))
            edge_num = edge_lookup[key]
            border_regions.append([edge_num, d2])
            border_regions.append([edge_num, d2 + 1])
        else:
            border_regions.append([face, d1 - 1, d2])
            border_regions.append([face, d1 - 1, d2 + 1])
        # Border for edge between c1 and c2.
        if d1 + d2 == edge_length - 1:
            key = tuple(sorted((c1, c2)))
            edge_num = edge_lookup[key]
            border_regions.append([edge_num, d2])
            border_regions.append([edge_num, d2 + 1])
        else:
            border_regions.append([face, d1 + 1, d2])
            border_regions.append([face, d1, d2 + 1])
    return border_regions

def region_to_position(region: tuple) -> np.ndarray:
    # Convert tuple back to list for internal processing.
    region_list = list(region)
    if len(region_list) == 1:
        return corner_positions[region_list[0]]
    elif len(region_list) == 2:
        edge_num, pos = region_list
        c0, c1 = edge_number_to_vertices[edge_num]
        # Choose one of the adjacent faces (from precomputed dictionary).
        key = tuple(sorted((c0, c1)))
        face, _ = edge_to_faces[key][0]
        if [c0, c1] == face_number_to_corners[face][:2]:
            matrix_coords = [pos, 0, 0]
        elif [c0, c1] == face_number_to_corners[face][1:]:
            matrix_coords = [edge_length - pos, pos, 0]
        else:
            matrix_coords = [0, pos, 0]
    else: # len(region_list) == 3
        face, d1, d2 = region_list
        matrix_coords = [d1, d2, 0]
    # Use precomputed transform.
    pos = np.dot(np.array(matrix_coords), face_transform[face]) + face_origin[face]
    return pos / np.linalg.norm(pos)

def point_in_triangle(P, tri):
    """
    Check if point P = (px, py) lies inside triangle tri = [(x1,y1), (x2,y2), (x3,y3)]
    Returns True if inside (including edges), False otherwise.
    """
    (x1, y1), (x2, y2), (x3, y3) = tri
    px, py = P

    # Compute signed areas (cross products)
    d1 = (px - x2) * (y1 - y2) - (py - y2) * (x1 - x2)
    d2 = (px - x3) * (y2 - y3) - (py - y3) * (x2 - x3)
    d3 = (px - x1) * (y3 - y1) - (py - y1) * (x3 - x1)

    # Check if all have same sign (or zero, meaning on the edge)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

def predict_round_offset(matrix_coords) -> list:
    # if matrix_coords fits into some rounding triangle then we will only return that round offset
    # The V1+ triangle has corners [1/2, 0], [1/3, 1/3], [1/2, 1/2]
    offsetted_coords = float(matrix_coords[0] - round(matrix_coords[0])), float(matrix_coords[1] - round(matrix_coords[1]))
    
    if point_in_triangle(offsetted_coords, [(1/2, 0), (1/3, 1/3), (1/2, 1/2)]):
        # V1+ triangle
        return (1, 0)

    if point_in_triangle(offsetted_coords, [(0, 1/2), (1/3, 1/3), (1/2, 1/2)]):
        # V2+ triangle
        return (0, 1)
        
    if point_in_triangle(offsetted_coords, [(-1/2, 0), (-1/3, -1/3), (-1/2, -1/2)]):
        # V1- triangle
        return (-1, 0)

    if point_in_triangle(offsetted_coords, [(0, -1/2), (-1/3, -1/3), (-1/2, -1/2)]):
        # V2- triangle
        return (0, -1)

    return (0, 0)

def get_offset_cheating(approx_region: list, point: list) -> list:
    candidates = [approx_region] + region_to_border_regions(approx_region)
    best_region = approx_region
    best_dist = float('inf')
    for candidate in candidates:
        # Cache key conversion (lists to tuples) for region_to_position.
        candidate_key = tuple(candidate)
        pos_candidate = region_to_position(candidate_key)
        d = (pos_candidate[0] - point[0]) ** 2 + (pos_candidate[1] - point[1]) ** 2 + (pos_candidate[2] - point[2]) ** 2
        if d < best_dist:
            best_dist = d
            best_region = candidate
    if best_region == approx_region:
        return best_region, (0, 0)
    elif len(best_region) == len(approx_region) == 3:
        return best_region, (best_region[1] - approx_region[1], best_region[2] - approx_region[2])
    else:
        # bloddy edge case just get it wrong
        return best_region, (0, 0)

def face_coords_to_region(face_coords: list, sorted_corners: list, face: int) -> list:
    if face_coords == [0, 0]:
        approx_region = [sorted_corners[0]]
    elif face_coords == [edge_length, 0]:
        approx_region = [sorted_corners[1]]
    elif face_coords == [0, edge_length]:
        approx_region = [sorted_corners[2]]
    elif face_coords[1] == 0:
        key = tuple(sorted((sorted_corners[0], sorted_corners[1])))
        approx_region = [edge_lookup[key], face_coords[0]]
    elif face_coords[0] == 0:
        key = tuple(sorted((sorted_corners[0], sorted_corners[2])))
        approx_region = [edge_lookup[key], face_coords[1]]
    elif face_coords[0] + face_coords[1] == edge_length:
        key = tuple(sorted((sorted_corners[1], sorted_corners[2])))
        approx_region = [edge_lookup[key], face_coords[1]]
    else:
        approx_region = [face, face_coords[0], face_coords[1]]
    
    return approx_region

def position_to_nearest_region(point: list | np.ndarray) -> list:
    point = np.array(point)
    # Vectorized computation for corner distances.
    dists = np.sum((corner_positions - point) ** 2, axis=1)
    best_three = np.argsort(dists)[:3]
    sorted_corners = sorted(best_three)
    # Lookup face based on the three closest corners.
    face = face_lookup.get(tuple(sorted_corners))
    if face is None:
        face = 0  # Fallback (should not occur)
    # Use precomputed inverse transform to compute matrix coordinates.
    c0 = face_origin[face]
    n_vec = face_transform[face][2]
    # Project the point onto the face plane.
    factor = np.dot(c0, n_vec) / np.dot(point, n_vec)
    p_proj = point * factor
    matrix_coords = np.dot(p_proj - c0, face_inv_transform[face])
    
    face_coords_raw = [round(matrix_coords[0]), round(matrix_coords[1])]

    predicted_offset = predict_round_offset(matrix_coords)
    face_coords = [face_coords_raw[0] + predicted_offset[0], face_coords_raw[1] + predicted_offset[1]]
    approx_region = face_coords_to_region(face_coords, sorted_corners, face)

    """
    This gets some coordintes on the face but its not quite accurate so we have to do this local check.
    Theoretically we should be able to skip it if we do the correction properly.
    Its not too much a performance drag but its a lot of extra code

    We only ever add or subract one to face_coords when we do the correction.
    we can check if the sum of the matrix_coords is less than or more than the sum of the face coords and that will tell us which side were on
    with the exeption of the spherical geometry quirk
    [2.49435395e+00 1.50286075e+00 1.11022302e-16]
    here on the sphere its closers to [3,1] but rounds to [2,2] which it is closest to post projection
    I dont imagine this will be a issue in practice
    """

   
    return approx_region