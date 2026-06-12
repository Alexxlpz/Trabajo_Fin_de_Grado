from datetime import datetime
from src.backend.repository.classes.Model import Model
from src.backend.repository.crop_repo import create_crop
from src.backend.repository.database import SessionLocal
from src.backend.repository.classes.Image import Image

def create_image_with_direction(path: str, latitude: float, longitude: float,
                                user_id: int, model_path: str,
                                num_sick: int = 0, num_healthy: int = 0,
                                upload_date: datetime | None = None,
                                crop_list: list | None = None,
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
            latitude=latitude,
            longitude=longitude,
            user_id=user_id
        )

        model = session.query(Model).filter(Model.path == model_path).first()

        if not model:
            model = Model(name=model_path.split("/")[-1], path=model_path)
            session.add(model)
            session.commit()
            session.refresh(model)

        img.model = model

        session.add(img)
        session.flush()
        for crop in (crop_list or []):
            create_crop(crop.status, crop.path, img.id, session)

        session.commit()
        session.refresh(img)
        return img
    except:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()
