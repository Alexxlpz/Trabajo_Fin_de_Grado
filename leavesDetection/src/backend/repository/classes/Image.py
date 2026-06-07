from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from src.backend.repository.database import Base

class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    num_sick = Column(Integer)
    num_healthy = Column(Integer)
    path = Column(String)
    upload_date = Column(DateTime)
    latitude = Column(Float)
    longitude = Column(Float)

    # Claves foráneas
    user_id = Column(Integer, ForeignKey("users.id"))
    model_id = Column(Integer, ForeignKey("models.id"))

    # Relaciones
    user = relationship("User", back_populates="images")
    model = relationship("Model", back_populates="image", uselist=False)
    crops = relationship("Crop", back_populates="image", cascade="all, delete-orphan")