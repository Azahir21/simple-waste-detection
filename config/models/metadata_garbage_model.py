from sqlalchemy import Column, BIGINT, Integer, TEXT, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from config.database import Base


class MetadataGarbage(Base):
    __tablename__ = "metadata_garbage"

    id = Column(BIGINT, primary_key=True, index=True)
    garbage_count = Column(Integer, nullable=True)
    garbage_detected = Column(Integer, nullable=True)
    garbage_names = Column(TEXT, nullable=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    files = relationship("File", backref="metadata_garbage")
