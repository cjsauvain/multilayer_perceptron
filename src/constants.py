BASE_FEATURES = (
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension"
)
SUFFIXES = ("_mean", "_se", "_worst")
FEATURES = tuple(f"{f}{s}" for s in SUFFIXES for f in BASE_FEATURES)
COLUMNS = ("ID", "diagnosis") + FEATURES
