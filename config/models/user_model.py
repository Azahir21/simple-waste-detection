from sqlalchemy import Column, BIGINT, VARCHAR, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from config.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(BIGINT, primary_key=True, index=True)
    username = Column(VARCHAR(65), unique=True, index=True)
    email = Column(VARCHAR(65), unique=True, index=True)
    password = Column(VARCHAR(255))
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    files = relationship("File", backref="user")
