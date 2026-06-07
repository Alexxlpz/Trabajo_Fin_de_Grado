from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from src.backend.repository.database import Base


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    path = Column(String)

    # Relaciones
    image = relationship("Image", back_populates="model", uselist=False)