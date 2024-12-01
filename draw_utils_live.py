import cv2
from utils import generate_description

color_map_live = {
"fan": (144, 238, 144),
"chair": (128, 0, 128),
"table": (255, 255, 0),
"monitor": (0, 128, 0),
"keyboard": (255, 165, 0),
"person": (255, 0, 0),
"door": (255, 192, 203),
}
def plot_boxes_live(results, frame, model, color_map_live):

    labels, descriptions = [], []
    for result in results:
        boxes = result.boxes.cpu().numpy()
        xyxys, confidences, class_ids = boxes.xyxy, boxes.conf, boxes.cls
        for xyxy, confidence, class_id in zip(xyxys, confidences, class_ids):
            label = model.names[int(class_id)]
            description = generate_description(label)  # Use generate_description here
            labels.append(label)
            descriptions.append(description)

            # Draw bounding box and label
            color = color_map_live.get(label, (0, 128, 0))
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, labels, descriptions
