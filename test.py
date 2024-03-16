from ultralytics import YOLO
model = YOLO('best.pt')
results = model(['rickshaw.jpg'])

for result in results:
    boxes = result.boxes
    result.show()
    result.save(filename = 'result.jpg')