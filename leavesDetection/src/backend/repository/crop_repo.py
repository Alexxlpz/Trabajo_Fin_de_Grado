from src.backend.repository.classes.Crop import Crop
from src.backend.repository.database import SessionLocal

def create_crop(status: str,
                path: str,
                image_id: int,
                session=None
                ) -> Crop:
    own_session = False
    if session is None:
        session = SessionLocal()
        own_session = True

    try:
        crop = Crop(
            status=status,
            path=path,
            image_id=image_id
        )

        session.add(crop)
        session.commit()
        session.refresh(crop)
        return crop
    except:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()
