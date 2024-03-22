from typing_extensions import Annotated
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from services.Auth import jwt_service
from config.schemas.commond_schema import TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/login", scheme_name="JWT")


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    service_jwt: jwt_service.JWTService = Depends(),
):
    return TokenData.parse_obj(service_jwt.decode_access_token(token))
