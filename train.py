from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def main():
    try:
        model.train(data='Dataset/SplitData/data.yaml', epochs=0)
    except Exception as e:
        print(str(e))

if __name__ == '__main__':
    main()