from sqlalchemy import Column, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship
from src.backend.repository.database import Base

class Direction(Base):
    __tablename__ = "directions"

    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float)
    longitude = Column(Float)

    # Vinculación con la imagen
    image_id = Column(Integer, ForeignKey("images.id"))
    image = relationship("Image", back_populates="direction")