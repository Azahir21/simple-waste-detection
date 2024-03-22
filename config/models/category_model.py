from sqlalchemy import Column, BIGINT, VARCHAR, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from config.database import Base


class Category(Base):
    __tablename__ = "categories"

    id = Column(BIGINT, primary_key=True, index=True)
    name = Column(VARCHAR(65), unique=True, index=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    files = relationship("File", backref="category")
