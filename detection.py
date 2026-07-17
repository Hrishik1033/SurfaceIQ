from ultralytics import YOLO

model = YOLO('models/model.pt')
#img_path = "C:/Users/HRISHIK BHATTACHARYA/Downloads/0025bde0c_jpg.rf.2094797342aeb5ca8951065e99a0b9bf.jpg"

defect_names = {'class-3':' Scratch / gouge / rolling scar',
                'class-4':'Surface flaking / patch defect',
                'class-1':'Hairline crack / fine linear mark',
                'class-2':'seam, edge crack, or roll-mark defect'}
def get_defect(img_path)->list:
    defects = []
    results = model.predict(img_path)
    for i, mask in enumerate(results[0].masks):
        cls_id = int(results[0].boxes.cls[i])
        cls_name = model.names[cls_id]
        confidence = float(results[0].boxes.conf[i])
        polygon_points = mask.xy[0]  # the actual polygon coordinates
        #Take the dictionary keys as the cls_names and values as the defect descriptions
        #Return only the list of defect names
        defects.append(defect_names[cls_name])
    return defects
#print(get_defect(img_path))