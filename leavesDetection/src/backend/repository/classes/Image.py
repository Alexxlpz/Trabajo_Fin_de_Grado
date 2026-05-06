from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from src.backend.repository.database import Base

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    num_sick = Column(Integer)
    num_healthy = Column(Integer)
    path = Column(String)
    upload_date = Column(DateTime) # todo no estaba puesto pero la fecha de subida puede ser util

    # Claves foráneas
    user_id = Column(Integer, ForeignKey("users.id"))

    # Relaciones
    user = relationship("User", back_populates="images")
    direction = relationship("Direction", back_populates="image", uselist=False)