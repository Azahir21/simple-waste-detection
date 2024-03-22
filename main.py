from fastapi import FastAPI
from routes.image_route import image_router
from routes.ui_route import ui_router
from routes.auth_route import auth_router
from config.database import engine, get_db
from fastapi.middleware.cors import CORSMiddleware
from config.models import (
    category_model,
    file_model,
    location_model,
    metadata_garbage_model,
    type_model,
    user_model,
)

app = FastAPI(debug=True)

origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

location_model.Base.metadata.create_all(bind=engine)
user_model.Base.metadata.create_all(bind=engine)
category_model.Base.metadata.create_all(bind=engine)
file_model.Base.metadata.create_all(bind=engine)
metadata_garbage_model.Base.metadata.create_all(bind=engine)
type_model.Base.metadata.create_all(bind=engine)

# Include the image router
app.include_router(auth_router)
app.include_router(image_router)
app.include_router(ui_router)
