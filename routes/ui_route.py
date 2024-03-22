import base64
from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.templating import Jinja2Templates

from services.image_service import ServiceImage


ui_router = APIRouter(tags=["UI"])
templates = Jinja2Templates(directory="templates")


def encode_image_to_base64(image_path):
    print(f"{image_path}")
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image


@ui_router.post("/prediction")
async def prediction(
    request: Request,
    file: UploadFile = File(...),
    service_image: ServiceImage = Depends(),
):
    predict = service_image.predict(file)
    original_image = predict["original_image"]
    result_image = predict["result_image"]
    object_count = predict["object_count"]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "original_image": encode_image_to_base64(original_image),
            "result_image": encode_image_to_base64(result_image),
            "object_count": object_count,
        },
    )


@ui_router.get("/")
async def user_interface(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
