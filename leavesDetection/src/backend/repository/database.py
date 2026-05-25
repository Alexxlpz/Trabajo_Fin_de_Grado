from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
#docker run --name leaves-db -e POSTGRES_PASSWORD=postgres-database -p 5432:5432 -d postgres
# ENCENDER DOCKER PRIMERO
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:postgres-database@localhost:5432/postgres"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def init_db():
    import src.backend.repository.classes.User  # noqa: F401
    import src.backend.repository.classes.Direction  # noqa: F401
    import src.backend.repository.classes.Image  # noqa: F401

    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()