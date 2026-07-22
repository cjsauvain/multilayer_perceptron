def build_column_names() -> list:
    features = [
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
    ]
    columns = [
        "ID",
        "diagnosis",
    ] + [
        feature + "_mean" for feature in features
    ] + [
        feature + "_se" for feature in features
    ] + [
        feature + "_worst" for feature in features
    ]

    return columns
