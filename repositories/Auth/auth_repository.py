from fastapi import Depends, HTTPException
from config.database import get_db
from config.models import user_model
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError


class AuthRepository:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db

    async def insert_new_user(self, user: user_model.User):
        try:
            new_user = user_model.User(**user.dict())
            self.db.add(new_user)
            self.db.commit()
            self.db.refresh(new_user)
            print("success: ", new_user)
            return new_user
        except SQLAlchemyError as e:
            raise HTTPException(status_code=500, detail="Database error")

    async def find_user_by_username(self, username: str):
        try:
            return (
                self.db.query(user_model.User)
                .filter(user_model.User.username == username)
                .first()
            )
        except SQLAlchemyError as e:
            raise HTTPException(status_code=500, detail="Database error")

    async def find_user_by_email(self, email: str):
        try:
            data = (
                self.db.query(user_model.User)
                .filter(user_model.User.email == email)
                .first()
            )
            return data
        except SQLAlchemyError as e:
            raise HTTPException(status_code=500, detail="Database error")
