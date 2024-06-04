import cv2
import asyncio
import websockets
import numpy as np
import ssl
import base64

# Load YOLO model
# download yolo weights from https://github.com/patrick013/Object-Detection---Yolov3/blob/master/model/yolov3.weights
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

async def send_video():
    cap = cv2.VideoCapture(0)
    ssl_context = ssl._create_unverified_context()
    async with websockets.connect('wss://localhost:7181/ws', ssl=ssl_context, ping_interval=60, ping_timeout=120) as websocket:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform YOLO object detection
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                # Initialization
                class_ids = []
                confidences = []
                boxes = []

                # For each detection from each output layer
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            # Object detected
                            center_x = int(detection[0] * frame.shape[1])
                            center_y = int(detection[1] * frame.shape[0])
                            w = int(detection[2] * frame.shape[1])
                            h = int(detection[3] * frame.shape[0])

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                # Draw bounding boxes on the frame
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        confidence = confidences[i]
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                #cv2.imshow('Video', frame)
                _, img_encoded = cv2.imencode('.png', frame)
                img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
                await websocket.send(img_base64.encode('utf-8'))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Connection closed with error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cap.release()

asyncio.get_event_loop().run_until_complete(send_video())