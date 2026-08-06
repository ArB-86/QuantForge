from __future__ import annotations


class TopicClassifier:
    """
    Rule-based topic classifier for CBSE Class 10 Mathematics.
    """

    TOPICS = {
        "Real Numbers": [
            "hcf", "lcm", "euclid", "euclidean",
            "fundamental theorem of arithmetic",
            "prime factorisation", "irrational", "terminating",
            "non-terminating"
        ],

        "Polynomials": [
            "polynomial", "zeroes", "zeros",
            "quadratic polynomial",
            "cubic polynomial",
            "remainder theorem",
            "factor theorem"
        ],

        "Pair of Linear Equations": [
            "pair of equations",
            "linear equations",
            "graphically",
            "intersecting",
            "parallel",
            "coincident",
            "unique solution",
            "infinitely many solutions",
            "no solution"
        ],

        "Quadratic Equations": [
            "quadratic equation",
            "quadratic",
            "roots of",
            "discriminant",
            "nature of roots"
        ],

        "Arithmetic Progressions": [
            "arithmetic progression",
            "ap",
            "common difference",
            "nth term",
            "sum of first"
        ],

        "Triangles": [
            "triangle",
            "triangles",
            "similar",
            "similarity",
            "pythagoras",
            "pythagorean"
        ],

        "Coordinate Geometry": [
            "distance formula",
            "section formula",
            "coordinates",
            "coordinate",
            "mid-point",
            "midpoint"
        ],

        "Introduction to Trigonometry": [
            "sin",
            "cos",
            "tan",
            "cot",
            "sec",
            "cosec",
            "trigonometric",
            "trigonometry"
        ],

        "Applications of Trigonometry": [
            "angle of elevation",
            "angle of depression",
            "tower",
            "height",
            "observer",
            "ladder"
        ],

        "Circles": [
            "circle",
            "radius",
            "diameter",
            "tangent",
            "chord"
        ],

        "Areas Related to Circles": [
            "sector",
            "segment",
            "arc",
            "circumference",
            "area of circle",
            "concentric"
        ],

        "Surface Areas and Volumes": [
            "cube",
            "cuboid",
            "sphere",
            "cone",
            "cylinder",
            "hemisphere",
            "frustum",
            "volume",
            "surface area"
        ],

        "Statistics": [
            "mean",
            "median",
            "mode",
            "frequency",
            "histogram",
            "ogive",
            "grouped data"
        ],

        "Probability": [
            "probability",
            "random",
            "event",
            "sample space",
            "outcome"
        ]
    }

    def classify(self, question):

        text = question.question.lower()

        best_topic = "Unknown"
        best_score = 0

        for topic, keywords in self.TOPICS.items():

            score = 0

            for keyword in keywords:

                if keyword in text:
                    score += 1

            if score > best_score:
                best_score = score
                best_topic = topic

        return best_topic
