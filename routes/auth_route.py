from typing_extensions import Annotated
from fastapi import Depends, HTTPException
from fastapi import APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from config.schemas.auth_schema import InputLogin, InputUser, OutputLogin
from config.schemas.commond_schema import StandardResponse, TokenData
from services.Auth.auth_service import AuthServices
from services.common_service import get_current_user


auth_router = APIRouter(prefix="/api/v1", tags=["Auth"])


@auth_router.post("/login")
async def user_login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    service_user: AuthServices = Depends(),
):
    jwt_token = await service_user.login_user(
        InputLogin(
            email=form_data.username,
            password=form_data.password,
        )
    )
    return OutputLogin(access_token=jwt_token, token_type="bearer")


@auth_router.post("/register")
async def user_register(input_user: InputUser, service_user: AuthServices = Depends()):
    await service_user.insert_new_user(input_user)
    return StandardResponse(detail="Success Register User")


@auth_router.get("/profile")
async def user_profile(
    token: Annotated[TokenData, Depends(get_current_user)],
    service_user: AuthServices = Depends(),
):
    return await service_user.get_current_user(token)
