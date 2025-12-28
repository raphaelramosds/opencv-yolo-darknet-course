import os

def traduzir_labels(classes_path, annotations_dir):
    """
    Converte labels de anotações de nomes de classes para números de classes conforme o arquivo de classes fornecido.
    """
    classes = {}
    with open(classes_path, "r") as myFile:
        for num, line in enumerate(myFile, 0):
            line = line.rstrip("\n")
            classes[line] = num
        myFile.close()
    
    os.chdir(annotations_dir)

    for filename in os.listdir(os.getcwd()):

        if filename.endswith(".txt"):

            annotations = []

            with open(filename, "r") as file:

                for line in file:
                    line = line.rstrip("\n")
                    parts = line.split(" ")
                    class_name = parts[0]
                    coords = list(map(float, parts[1:5]))
                    if class_name in classes:
                        class_num = classes[class_name]
                        annotations.append(f"{class_num} {' '.join(map(str, coords))}\n")
                file.close()
            
            with open(filename, "w") as outfile:
                for line in annotations:
                    outfile.write(line)
                outfile.close()

if __name__ == "__main__":

    base_path = "/home/rapha/Learn/opencv-yolo-darknet-course-backup/shared/dataset"
    classes_path = f"{base_path}/obj.names"
    annotations_dir = f"{base_path}/data/val"

    traduzir_labels(classes_path, annotations_dir)