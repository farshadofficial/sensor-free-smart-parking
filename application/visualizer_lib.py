 # Imports
import cv2
import numpy as np
import json

# Structure holding variables being tracked from each image
class ImageData:
    def __init__(self, image, filename=None, point_data_path=None):
        self.image = image
        self.entry_points = []
        self.occupied_spaces = []
        self.vacant_spaces = []

        if image is None:
            if filename:
                self.image = cv2.imread(filename)
            if self.image is None:
                # Generate a white 1000x1000 image with text indicating no image data
                self.image = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
                cv2.putText(self.image, 'No Image Data', (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                print("Warning: No image data provided. Generated placeholder image.")

        self.width = self.image.shape[1] if self.image is not None else 0
        self.height = self.image.shape[0] if self.image is not None else 0

        if point_data_path:
            self.load_point_data(point_data_path)

    def load_point_data(self, point_data_path):
        # Load point data from a JSON file
        with open(point_data_path, 'r') as f:
            point_data = json.load(f)
            self.entry_points = point_data.get('entry_points', [])
            self.occupied_spaces = point_data.get('occupied_spaces', [])
            self.vacant_spaces = point_data.get('vacant_spaces', [])

    # Select entry point based on user input (all others are un-selected)
    def select_entry_point(self, selected_entry_point):
        for entry_point in self.entry_points:
            entry_point['selected'] = (entry_point == selected_entry_point)

    # Get occupancy status of the image
    def get_occupancy_status(self):
        return {
            "occupied_spaces": self.occupied_spaces,
            "vacant_spaces": self.vacant_spaces,
            "occupancy_rate": len(self.occupied_spaces) / (len(self.occupied_spaces) + len(self.vacant_spaces)) if (len(self.occupied_spaces) + len(self.vacant_spaces)) > 0 else 0
        }

# Get the closest vacant space to a given entry point
def get_closest_vacant_space(image_data, entry_point_coordinates):
    closest_space = None
    min_distance = float('inf')

    for space in image_data.vacant_spaces:
        space_coordinates = space['coordinates']
        distance = ((entry_point_coordinates[0] - space_coordinates[0]) ** 2 + (entry_point_coordinates[1] - space_coordinates[1]) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_space = space

    return closest_space

    # Visualize the parking spaces over the image, and label them as occupied or vacant


def visualize_parking_spaces(image_data, show_occupancy_rate=False, show_closest_vacancy=False):
    # Visualize occupied spaces in red
    if (image_data is None or image_data.image is None):
        # Generate a white 1000x1000 image with text indicating no image data
        image_data = ImageData(np.ones((1000, 1000, 3), dtype=np.uint8) * 255)
        cv2.putText(image_data.image, 'No Image Data', (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    for space_data in image_data.occupied_spaces:
        coordinates = space_data['coordinates']
        size = space_data['size']
        cv2.rectangle(image_data.image, (coordinates[0], coordinates[1]),
                      (coordinates[0] + size[0], coordinates[1] + size[1]), (0, 0, 255), 2)
        cv2.putText(image_data.image, 'Occupied', (coordinates[0], coordinates[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)

    closest_vacant_space = None

    if (show_closest_vacancy):
        # Iterate through entry points and find the first one tagged as "selected"
        for entry_point in image_data.entry_points:
            if entry_point.get('selected', True):
                closest_vacant_space = get_closest_vacant_space(image_data, entry_point['coordinates'])
                if closest_vacant_space:
                    # Highlight the entry point in green
                    cv2.circle(image_data.image, (entry_point['coordinates'][0], entry_point['coordinates'][1]), 5,
                               (0, 255, 0), -1)
                    # Highlight the closest vacant space in green
                    coordinates = closest_vacant_space['coordinates']
                    size = closest_vacant_space['size']
                    cv2.rectangle(image_data.image, (coordinates[0], coordinates[1]),
                                  (coordinates[0] + size[0], coordinates[1] + size[1]), (0, 255, 0), 2)
                    cv2.putText(image_data.image, 'Closest Vacant', (coordinates[0], coordinates[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                break

    # Visualize vacant spaces in blue
    for space_data in image_data.vacant_spaces:
        if show_closest_vacancy and closest_vacant_space and space_data == closest_vacant_space:
            continue  # Skip the closest vacant space since it's already highlighted
        coordinates = space_data['coordinates']
        size = space_data['size']
        cv2.rectangle(image_data.image, (coordinates[0], coordinates[1]),
                      (coordinates[0] + size[0], coordinates[1] + size[1]), (255, 0, 0), 2)
        cv2.putText(image_data.image, 'Vacant', (coordinates[0], coordinates[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 1)

    # Visualize entry points in yellow
    for entry_point in image_data.entry_points:
        if show_closest_vacancy and closest_vacant_space and entry_point.get('selected', True):
            continue  # Skip the selected entry point since it's already highlighted
        cv2.circle(image_data.image, (entry_point['coordinates'][0], entry_point['coordinates'][1]), 5, (0, 255, 255),
                   -1)
        cv2.putText(image_data.image, 'Entry Point',
                    (entry_point['coordinates'][0] + 10, entry_point['coordinates'][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)

    # Add a blob in the top-left corner showing the occupancy rate
    if (show_occupancy_rate):
        occupancy_status = image_data.get_occupancy_status()
        occupancy_rate_text = f"Occupancy Rate: {occupancy_status['occupancy_rate']:.2%}"
        cv2.rectangle(image_data.image, (10, 10), (200, 40), (255, 255, 255), -1)
        cv2.putText(image_data.image, occupancy_rate_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image_data.image

def external_visualize(img_path, point_path, show_occupancy_rate=False, show_closest_vacancy=False):
    image_data = ImageData(None, filename=img_path, point_data_path=point_path)
    if not image_data:
        print("No Image Data")
        return None
    if image_data.entry_points:
        image_data.select_entry_point(image_data.entry_points[0])
    visualized_image = visualize_parking_spaces(image_data, show_occupancy_rate=show_occupancy_rate, show_closest_vacancy=show_closest_vacancy)
    return visualized_image

# Example usage
if __name__ == "__main__":
    # Load image and point data from "../data" folder
    img_file = "../Data/val_8.jpg" # file = "../data/parking_lot_image.jpg"
    point_file = "../Data/parking_prediction_val_8.jpg.json"
    image_data = ImageData(None, filename=img_file, point_data_path=point_file)
    if not image_data:
        print("No Image Data")
    # Select the first entry point as the selected one
    if image_data.entry_points:
        image_data.select_entry_point(image_data.entry_points[0])
    # Visualize the parking spaces with occupancy rate and closest vacancy highlighted
    visualized_image = visualize_parking_spaces(image_data, show_occupancy_rate=True, show_closest_vacancy=True)
    # Display the visualized image, and allow for keyboard or GUI-based exit
    try:
        print(f"Size of image: {visualized_image.shape[1]}x{visualized_image.shape[0]}")
        cv2.imshow("Parking Lot Visualization", visualized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except KeyboardInterrupt or InterruptedError:
        cv2.destroyAllWindows()