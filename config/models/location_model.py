from sqlalchemy import Column, BIGINT, TEXT, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from config.database import Base
from geoalchemy2.types import Geometry


class Location(Base):
    __tablename__ = "locations"

    id = Column(BIGINT, primary_key=True, index=True)
    address = Column(TEXT, nullable=True)
    geo_location = Column(
        Geometry(
            geometry_type="POINT",
            srid=4326,
            spatial_index=False,
        )
    )
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    files = relationship("File", backref="location")
