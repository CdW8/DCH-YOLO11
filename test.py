from ultralytics import YOLO

def validate_model():

    model_path = "path/to/your/best.pt"
    data_path = "path/to/your/dataset.yaml"

    model = YOLO(model_path)
    metrics = model.val(
        data=data_path,
        task="detect",
        split="test",
        save=True,
        save_json=True
    )
    print("Validation finished. Metrics:")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50 : {metrics.box.map50:.4f}")
    print(f"mAP75 : {metrics.box.map75:.4f}")

if __name__ == "__main__":
    validate_model()
