from sqlalchemy import Column, Integer, String
from src.backend.repository.database import Base

class User(Base): # TODO realmente necesitamos informacion de la clase usuario?, quizas solo con la imagen
                  # y la direccion es suficiente
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)

    #images = relationship("Image", back_populates="user")