from fastapi import APIRouter, Depends, UploadFile, File
from services.image_service import ServiceImage
from fastapi.templating import Jinja2Templates


image_router = APIRouter(prefix="/api/v1", tags=["Image"])
templates = Jinja2Templates(directory="templates")


@image_router.post("/image")
def insert_new_image(
    file: UploadFile = File(...),
    service_image: ServiceImage = Depends(),
):
    return service_image.insert_image(file)


@image_router.get("/image/{filename}")
def download_image(
    filename: str,
    service_image: ServiceImage = Depends(),
):
    return service_image.download_image(f"assets/{filename}")


@image_router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    service_image: ServiceImage = Depends(),
):
    return service_image.predict(file)


@image_router.get("/")
async def root():
    return {"message": "Hello World"}
