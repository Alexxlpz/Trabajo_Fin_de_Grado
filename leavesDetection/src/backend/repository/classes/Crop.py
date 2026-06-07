from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from src.backend.repository.database import Base


class Crop(Base):
    __tablename__ = "crops"

    id = Column(Integer, primary_key=True, index=True)
    status = Column(String)
    path = Column(String)

    # Claves foráneas
    image_id = Column(Integer, ForeignKey("images.id"))

    # Relaciones
    image = relationship("Image", back_populates="crops")