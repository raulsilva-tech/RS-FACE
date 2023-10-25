from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def main():
    try:
        # model.train(data='Dataset/SplitData/dataOffline.yaml', epochs=20)
        model.train(data='NewDataset/data.yaml', epochs=20)
    except Exception as e:
        print(str(e))

if __name__ == '__main__':
    main()