import cv2
from utils import generate_description

color_map = {
"fan": (144, 238, 144),
"chair": (128, 0, 128),
"table": (255, 255, 0),
"monitor": (0, 128, 0),
"keyboard": (255, 165, 0),
"person": (255, 0, 0),
"door": (255, 192, 203),
}
def plot_boxes(results, frame, model, color_map):
    detection_data = []
    for result in results:
        # Extract bounding boxes, class IDs, and confidences
        boxes = result.boxes.cpu().numpy()
        xyxys, confidences, class_ids = boxes.xyxy, boxes.conf, boxes.cls
        
        # Process each detection
        for xyxy, confidence, class_id in zip(xyxys, confidences, class_ids):
            label = model.names[int(class_id)]
            detection_data.append({
                "Object": label,
                "Location (x, y)": (int(xyxy[0]), int(xyxy[1])),
                "Size (Width x Height)": (int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])),
                "Confidence": round(float(confidence), 2),
            })
            
            # Draw bounding box and label on the image/frame
            color = color_map.get(label, (0, 128, 0))  # Default to green if no color specified
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, detection_data

