import numpy as np
import math
import argparse

corner_number_to_position = [
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
]

# Edge definitions (30 edges connecting the vertices)
edge_number_to_vertices = [
    [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
    [1, 2], [2, 3], [3, 4], [4, 5], [1, 5],
    [1, 10], [1, 6], [2, 6], [2, 7], [3, 7], [3, 8], [4, 8], [4, 9], [5, 9], [5, 10],
    [6, 7], [7, 8], [8, 9], [9, 10], [6, 10],
    [6, 11], [7, 11], [8, 11], [9, 11], [10, 11]
]

# Face definitions (20 triangular faces)
face_number_to_corners = [
    [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 1, 5],
    [1, 2, 6], [2, 3, 7], [3, 4, 8], [4, 5, 9], [1, 5, 10],
    [1, 6, 10], [2, 6, 7], [3, 7, 8], [4, 8, 9], [5, 9, 10],
    [6, 7, 11], [7, 8, 11], [8, 9, 11], [9, 10, 11], [6, 10, 11]
]
'''print("{", end="")
for edge in edge_number_to_vertices:
    p0 = corner_number_to_position[edge[0]]
    p1 = corner_number_to_position[edge[1]]
    print(f"CircularArc((0,0,0), ({p0[0]:.8f}, {p0[1]:.8f}, {p0[2]:.8f}), ({p1[0]:.8f}, {p1[1]:.8f}, {p1[2]:.8f}))",
          end="," if edge != edge_number_to_vertices[-1] else "")
print("}")'''
# ---------- basic vector utilities ----------

def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)

def spherical_excess(a, b, c):
    """
    Oriented spherical excess (area) of triangle (a,b,c) on unit sphere.
    """
    T = np.dot(a, np.cross(b, c))
    D = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    return 2.0 * math.atan2(T, D)

# ---------- plane for fixed spherical area ----------

def area_plane(u, v, E):
    """
    Returns (n, h) such that n·x = h is the plane
    enforcing spherical area E(u,v,x) = E.
    """
    p = np.cross(u, v)
    d = u + v
    g = 1.0 + np.dot(u, v)
    k = math.tan(E / 2.0)
    n = p - k * d
    h = k * g
    return n, h

# ---------- solve X from area barycentric weights ----------

def area_barycentric_point(A, B, C, wA, wB, wC):
    """
    Compute point X on the unit sphere with spherical area barycentrics
    (wA, wB, wC) w.r.t. triangle (A,B,C).
    """
    A, B, C = map(normalize, (A, B, C))

    # total area (ensure positive orientation)
    E = spherical_excess(A, B, C)
    if E < 0:
        B, C = C, B
        E = -E

    # target sub-areas
    EA = wA * E  # E(XBC)
    EB = wB * E  # E(AXC)
    EC = wC * E  # E(ABX)

    # planes
    n1, h1 = area_plane(B, C, EA)
    n2, h2 = area_plane(C, A, EB)
    n3, h3 = area_plane(A, B, EC)

    # solve 3x3 linear system
    N = np.vstack([n1, n2, n3])
    h = np.array([h1, h2, h3])

    X = np.linalg.solve(N, h)

    # normalize (floating-point safety)
    return normalize(X)

# ---------- generate barycentric grid ----------

def spherical_area_barycentric_grid(A, B, C, N):
    """
    Generate all spherical area-barycentric grid points
    with N subdivisions per side.
    Returns list of (point, (wA, wB, wC)) tuples.
    """
    points = []
    for i in range(N + 1):
        for j in range(N + 1 - i):
            k = N - i - j
            wA = i / N
            wB = j / N
            wC = k / N
            X = area_barycentric_point(A, B, C, wA, wB, wC)
            points.append((X, (wA, wB, wC)))
    return points

# ---------- gnomonic projection grid ----------

def gnomonic_barycentric_grid(A, B, C, N):
    """
    Generate grid points using planar (linear) barycentric interpolation,
    then project outward onto the unit sphere (gnomonic projection).
    Returns list of (point, (wA, wB, wC)) tuples.
    """
    A, B, C = map(normalize, (np.asarray(A), np.asarray(B), np.asarray(C)))
    points = []
    for i in range(N + 1):
        for j in range(N + 1 - i):
            k = N - i - j
            wA = i / N
            wB = j / N
            wC = k / N
            # Linear barycentric interpolation on the plane
            P = wA * A + wB * B + wC * C
            # Project outward onto unit sphere
            X = normalize(P)
            points.append((X, (wA, wB, wC)))
    return points

# ---------- naming helper ----------

def get_point_name(prefix, face_idx, face_corners, weights):
    """
    Generate point name based on prefix, face, and weights.
    If it's a corner (one weight is 1), use corner index.
    Otherwise use face and weights.
    """
    wA, wB, wC = weights
    # Check if it's a corner (one weight is 1, others are 0)
    if abs(wA - 1.0) < 1e-9 and abs(wB) < 1e-9 and abs(wC) < 1e-9:
        return f"{prefix}_corner_{face_corners[0]}"
    elif abs(wB - 1.0) < 1e-9 and abs(wA) < 1e-9 and abs(wC) < 1e-9:
        return f"{prefix}_corner_{face_corners[1]}"
    elif abs(wC - 1.0) < 1e-9 and abs(wA) < 1e-9 and abs(wB) < 1e-9:
        return f"{prefix}_corner_{face_corners[2]}"
    else:
        # Format weights nicely (remove trailing zeros)
        def fmt_w(w):
            if w == int(w):
                return str(int(w))
            return f"{w:.4f}".rstrip('0').rstrip('.')
        return f"{prefix}_face_{face_idx}_w{fmt_w(wA)}_{fmt_w(wB)}_{fmt_w(wC)}"

def main(args_list):
    N = args_list[0]
    output_file = args_list[1]
    face = args_list[2]
    spherical = args_list[3]


    # Determine which faces to process
    if face.lower() == "all":
        face_indices = list(range(len(face_number_to_corners)))
        face_desc = "all 20 faces"
    else:
        face_idx = int(face)
        if face_idx < 0 or face_idx >= len(face_number_to_corners):
            print(f"Error: Face index must be 0-{len(face_number_to_corners)-1}")
            exit(1)
        face_indices = [face_idx]
        face_desc = f"face {face_idx}"

    # collect gnomonic projection points (default)
    # Each entry: (point, face_idx, face_corners, weights)
    gnomonic_pts = []
    for face_idx in face_indices:
        face = face_number_to_corners[face_idx]
        A = corner_number_to_position[face[0]]
        B = corner_number_to_position[face[1]]
        C = corner_number_to_position[face[2]]
        pts = gnomonic_barycentric_grid(A, B, C, N)
        for pt, weights in pts:
            gnomonic_pts.append((pt, face_idx, face, weights))

    # collect spherical area barycentric points if enabled
    spherical_pts = []
    if spherical:
        for face_idx in face_indices:
            face = face_number_to_corners[face_idx]
            A = corner_number_to_position[face[0]]
            B = corner_number_to_position[face[1]]
            C = corner_number_to_position[face[2]]
            pts = spherical_area_barycentric_grid(A, B, C, N)
            for pt, weights in pts:
                spherical_pts.append((pt, face_idx, face, weights))

    # Convert unit sphere (x, y, z) to lat/lon
    def xyz_to_latlon(x, y, z):
        lat = math.degrees(math.asin(z))
        lon = math.degrees(math.atan2(y, x))
        return lat, lon

    # Build KML file
    spherical_desc = " + spherical" if spherical else ""
    kml_header = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>Grid Points ({face_desc}, N={N}{spherical_desc})</name>
  <Style id="sphericalStyle">
    <IconStyle>
      <scale>5</scale>
      <color>ff0000ff</color>
      <Icon>
        <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>
      </Icon>
    </IconStyle>
  </Style>
  <Style id="gnomonicStyle">
    <IconStyle>
      <scale>5</scale>
      <color>ff00ff00</color>
      <Icon>
        <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>
      </Icon>
    </IconStyle>
  </Style>
'''
    
    kml_footer = '''</Document>
</kml>'''

    # Build gnomonic folder (green) - default
    gnomonic_placemarks = []
    seen_names_g = set()
    for pt, face_idx, face_corners, weights in gnomonic_pts:
        name = get_point_name("G", face_idx, face_corners, weights)
        # Skip duplicate corners (same corner appears in multiple faces)
        if name in seen_names_g:
            continue
        seen_names_g.add(name)
        lat, lon = xyz_to_latlon(pt[0], pt[1], pt[2])
        gnomonic_placemarks.append(f'''    <Placemark>
      <name>{name}</name>
      <styleUrl>#gnomonicStyle</styleUrl>
      <Point>
        <coordinates>{lon:.8f},{lat:.8f},0</coordinates>
      </Point>
    </Placemark>''')

    gnomonic_folder = f'''  <Folder>
    <name>Gnomonic Projection Points ({len(gnomonic_placemarks)})</name>
{chr(10).join(gnomonic_placemarks)}
  </Folder>'''

    # Build spherical barycentric folder (red) - optional
    spherical_folder = ""
    spherical_placemarks = []
    if spherical_pts:
        seen_names = set()
        for pt, face_idx, face_corners, weights in spherical_pts:
            name = get_point_name("S", face_idx, face_corners, weights)
            # Skip duplicate corners (same corner appears in multiple faces)
            if name in seen_names:
                continue
            seen_names.add(name)
            lat, lon = xyz_to_latlon(pt[0], pt[1], pt[2])
            spherical_placemarks.append(f'''    <Placemark>
      <name>{name}</name>
      <styleUrl>#sphericalStyle</styleUrl>
      <Point>
        <coordinates>{lon:.8f},{lat:.8f},0</coordinates>
      </Point>
    </Placemark>''')

        spherical_folder = f'''
  <Folder>
    <name>Spherical Barycentric Points ({len(spherical_placemarks)})</name>
{chr(10).join(spherical_placemarks)}
  </Folder>'''

    kml_content = kml_header + gnomonic_folder + spherical_folder + "\n" + kml_footer

    # Write to KML file
    with open(output_file, "w") as f:
        f.write(kml_content)

    print(f"Generated {len(gnomonic_placemarks)} gnomonic projection points from {face_desc} with N={N}")
    if spherical:
        print(f"Generated {len(spherical_placemarks)} spherical barycentric points")
    print(f"✓ Saved to {output_file}")
# ---------- main ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate spherical grid points on icosahedron faces and save to KML."
    )
    parser.add_argument(
        "-n", "--subdivisions",
        type=int,
        default=6,
        help="Number of subdivisions per side (default: 6)"
    )
    parser.add_argument(
        "-f", "--face",
        type=str,
        default="all",
        help="Face index (0-19) or 'all' for all faces (default: all)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.kml",
        help="Output KML filename (default: output.kml)"
    )
    parser.add_argument(
        "-s", "-spherical",
        type=lambda x: x.lower() == 'true',
        default=False,
        help="Include spherical barycentric points (default: false)"
    )
    
    args = parser.parse_args()
    args_list = [args.subdivisions, args.output, args.face, args.spherical]
    main(args_list)