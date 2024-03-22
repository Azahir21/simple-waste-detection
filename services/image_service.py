import datetime
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse
import subprocess
from ultralytics import YOLO

model = YOLO("../best.pt")


class ServiceImage:
    def __init__(self) -> None:
        pass

    def insert_image(self, file: UploadFile):
        try:
            content = file.file.read()
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{file.filename}"
            with open(f"assets/{filename}", "wb") as f:
                f.write(content)
        except Exception as e:
            print(e)
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()
        return filename

    def download_image(self, filename: str):
        try:
            with open(filename, "rb") as f:
                content = f.read()
        except Exception as e:
            print(e)
            return {"message": "There was an error downloading the file"}
        return FileResponse(
            f"{filename}",
            media_type="application/octet-stream",
            filename=filename,
        )

    def predict(self, file: UploadFile):
        image_path = self.insert_image(file)
        path = f"assets/{image_path}"
        command = f"yolo task=segment mode=predict model='best.pt' source={path} name='yolov8s_predict' exist_ok=True save=True save_txt=True"
        subprocess.run(command, shell=True)
        labels_filename = (
            f"runs/segment/yolov8s_predict/labels/{image_path.split('.')[0]}.txt"
        )
        object_count = self.object_count(labels_filename)
        return {
            "original_image": path,
            "result_image": f"runs/segment/yolov8s_predict/{image_path}",
            "object_count": object_count,
        }

        # return (self.download_image(f"runs/segment/yolov8s_predict/{image_path}"),)

    def object_count(self, filename: str):
        with open(filename, "r") as file:
            lines = file.readlines()
        first_values = [float(line.split()[0]) for line in lines]
        label_mapping = {
            0: "rov",
            1: "plant",
            2: "animal_fish",
            3: "animal_starfish",
            4: "animal_shells",
            5: "animal_crab",
            6: "animal_eel",
            7: "animal_etc",
            8: "trash_etc",
            9: "trash_fabric",
            10: "trash_fishing_gear",
            11: "trash_metal",
            12: "trash_paper",
            13: "trash_plastic",
            14: "trash_rubber",
            15: "trash_wood",
        }
        labels = [label_mapping[value] for value in first_values]
        count_dict = {}
        for label in labels:
            count_dict[label] = count_dict.get(label, 0) + 1
        return count_dict
