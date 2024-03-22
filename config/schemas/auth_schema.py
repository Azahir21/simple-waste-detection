from pydantic import BaseModel


class InputUser(BaseModel):
    username: str
    email: str
    password: str


class InputLogin(BaseModel):
    email: str
    password: str


class OutputProfile(BaseModel):
    username: str
    email: str


class OutputLogin(BaseModel):
    access_token: str
    token_type: str
