from datetime import datetime
from src.backend.repository.database import SessionLocal
from src.backend.repository.classes.Image import Image
from src.backend.repository.classes.Direction import Direction

def create_image_with_direction(path: str, latitude: float, longitude: float,
                                user_id: int,
                                num_sick: int = 0, num_healthy: int = 0,
                                upload_date: datetime | None = None,
                                session=None) -> Image:
    own_session = False
    if session is None:
        session = SessionLocal()
        own_session = True

    upload_date_formated = None
    if upload_date is not None:
        upload_date_formated = datetime.strptime(str(upload_date), '%d/%m/%Y %H:%M:%S')
    try:
        img = Image(
            path=path,
            num_sick=num_sick,
            num_healthy=num_healthy,
            upload_date=upload_date_formated,
            user_id=user_id
        )
        dir = Direction(latitude=latitude, longitude=longitude)
        img.direction = dir  # vincula bidireccionalmente

        session.add(img)
        session.commit()
        session.refresh(img)
        return img
    except:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()
