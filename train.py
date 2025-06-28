from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("path/to/your/model.yaml")
    results = model.train(
        data="path/to/your/data.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
        pretrained=False
    )
    print(results)
