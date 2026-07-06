"""Utilities to export parking spot predictions to the JSON format
expected by the visualizer notebook.

Provides a single function with the same signature as
`visualize_parking_prediction(...)` from the project notebook so it can be
imported there. The function will write a JSON file containing:

- a placeholder entry point at (0, 0) marked selected
- `occupied_spaces` and `vacant_spaces` arrays using the top-left corner as
  `coordinates` and `[width, height]` as `size` (both integers).

The implementation is flexible about the input box format: it accepts each
spot as one of the following (per-spot):
- list/tuple [x, y, w, h] (top-left + size)
- list/tuple [x1, y1, x2, y2] (two corners)
- dict with keys 'x','y','w','h' or 'x1','y1','x2','y2'

The function also tolerates common predicted label encodings (0/1, True/False,
strings like "occupied"/"vacant"/"car").

Example usage:
    from export_prediction import export_parking_prediction_json
    export_parking_prediction_json(img_name, spots, true_labels, predicted_labels, model)

"""
import json
from typing import Iterable, List, Tuple, Union, Dict, Any, Optional

Box = Union[List[float], Tuple[float, ...], Dict[str, float]]

def _normalize_box(box: Box) -> Tuple[int, int, int, int]:
    """Return (x, y, w, h) as integers using the top-left corner as (x,y).

    Accepts the following per-box formats:
    - [x, y, w, h]
    - [x1, y1, x2, y2]
    - {'x':..,'y':..,'w':..,'h':..}
    - {'x1':..,'y1':..,'x2':..,'y2':..}
    """
    if isinstance(box, dict):
        if {'x', 'y', 'w', 'h'}.issubset(box.keys()):
            x = int(round(box['x']))
            y = int(round(box['y']))
            w = int(round(box['w']))
            h = int(round(box['h']))
            return x, y, w, h
        if {'x1', 'y1', 'x2', 'y2'}.issubset(box.keys()):
            x1 = float(box['x1'])
            y1 = float(box['y1'])
            x2 = float(box['x2'])
            y2 = float(box['y2'])
            x = int(round(min(x1, x2)))
            y = int(round(min(y1, y2)))
            w = int(round(abs(x2 - x1)))
            h = int(round(abs(y2 - y1)))
            return x, y, w, h
        raise ValueError("Unsupported box dict format: expected keys x,y,w,h or x1,y1,x2,y2")

    if isinstance(box, (list, tuple)) and len(box) == 4:
        a, b, c, d = box
        # Heuristic: if c > a and d > b and c-a and d-b are positive, treat as x1,y1,x2,y2
        try:
            a_f, b_f, c_f, d_f = float(a), float(b), float(c), float(d)
        except Exception:
            raise ValueError("Box values must be numeric")

        if c_f > a_f and d_f > b_f and (c_f - a_f) > 1 and (d_f - b_f) > 1:
            # likely x1,y1,x2,y2
            x = int(round(a_f))
            y = int(round(b_f))
            w = int(round(c_f - a_f))
            h = int(round(d_f - b_f))
            return x, y, w, h
        else:
            # treat as x,y,w,h
            x = int(round(a_f))
            y = int(round(b_f))
            w = int(round(c_f))
            h = int(round(d_f))
            return x, y, w, h

    raise ValueError("Unsupported box format. Expected list/tuple of length 4 or dict.")


def _is_occupied_label(label: Any) -> bool:
    """Return True if label indicates an occupied spot.

    Accepts numeric labels (1/0), booleans, and common strings.
    """
    if label is None:
        return False
    if isinstance(label, bool):
        return bool(label)
    if isinstance(label, (int, float)):
        return int(label) != 0
    if isinstance(label, str):
        s = label.strip().lower()
        return s in {"occupied", "car", "vehicle", "1", "true", "occupied_car"}
    return False

def _get_from_cars(car_boxes):
    occupied = []
    for car in car_boxes:
        x1, y1, x2, y2 = map(int, car)
        w, h = x2 - x1, y2 - y1
        occupied.append({"coordinates": [x1, y1], "size": [w, h]})
    return occupied


def export_parking_prediction_json(img_name: str,
                                   car_boxes,
                                   spots: Iterable[Box],
                                   true_labels: Optional[Iterable[Any]],
                                   predicted_labels: Iterable[Any],
                                   model: Any,
                                   image_folder: str = "parking_yolo/images/val",
                                   out_path: Optional[str] = None) -> str:
    """Generate a JSON file matching the visualizer schema from prediction data.

    Parameters:
    - img_name: name of the image (kept for potential future use)
    - spots: iterable of bounding boxes for parking spaces (see _normalize_box)
    - true_labels: ground-truth labels (not used but accepted for signature
      compatibility)
    - predicted_labels: predicted labels aligned with `spots` (used to decide
      occupied vs vacant)
    - model: model object (not used, accepted for signature compatibility)
    - image_folder: (optional) folder where the image lives (unused)
    - out_path: optional path to write JSON to. If None, defaults to
      `Visualizer/parking_prediction_{img_name}.json` under the project folder.

    Returns the path to the written JSON file.
    """
    spots_list = list(spots)
    preds_list = list(predicted_labels)

    occupied = _get_from_cars(car_boxes)
    vacant = []

    for spot, pred_label in zip(spots_list, preds_list):
        x, y, w, h = _normalize_box(spot)
        entry = {"coordinates": [int(x), int(y)], "size": [int(w), int(h)]}
        if not _is_occupied_label(pred_label):
            vacant.append(entry)

    data = {
        "entry_points": [
            {"coordinates": [0, 0], "selected": True}
        ],
        "occupied_spaces": occupied,
        "vacant_spaces": vacant
    }

    if out_path is None:
        # Create a "Data" folder in current working directory and place file there
        import os
        if not os.path.exists("Data"):
            os.makedirs("Data")
        out_path = f"Data/parking_prediction_{img_name}.json"

    # Ensure integers in JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return out_path


if __name__ == "__main__":
    # Quick self-test to generate a small example file when run directly
    sample_car_boxes = [
        [300, 200, 380, 320],  # x1,y1,x2,y2 detected car
    ]
    sample_spots = [
        [100, 200, 80, 120],
        [200, 200, 80, 120],
        [300, 200, 380, 320],  # demonstrates x1,y1,x2,y2 format
    ]
    sample_preds = [1, 0, "occupied"]
    out = export_parking_prediction_json("example", sample_car_boxes, sample_spots, None, sample_preds, None)
    print("Wrote:", out)

