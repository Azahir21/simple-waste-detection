from sqlalchemy import Column, BIGINT, TEXT, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from config.database import Base
from geoalchemy2 import Geometry


class File(Base):
    __tablename__ = "files"

    id = Column(BIGINT, primary_key=True, index=True)
    user_id = Column(BIGINT, ForeignKey("users.id"))
    type_id = Column(BIGINT, ForeignKey("types.id"))
    category_id = Column(BIGINT, ForeignKey("categories.id"))
    metadata_garbage_id = Column(BIGINT, ForeignKey("metadata_garbage.id"))
    location_id = Column(BIGINT, ForeignKey("locations.id"))
    description = Column(TEXT, nullable=True)
    caption = Column(TEXT, nullable=True)
    path = Column(TEXT, nullable=True)
    capture_date = Column(DateTime)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
