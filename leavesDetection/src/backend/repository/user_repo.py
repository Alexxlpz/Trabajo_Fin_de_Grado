import base64

from pydantic import EmailStr
from sqlalchemy.orm import Session
from sqlalchemy import desc
from src.backend.repository.classes.User import User
from src.backend.repository.classes.Image import Image


def authenticate_and_get_recent_paths(db: Session, email: EmailStr, password: str):
    user = db.query(User).filter(User.email == email).first()

    if not user or user.password != password:
        return False, [], -1

    print(user.id, user.username, user.email)

    recents = get_recent_paths(user.id, db)

    return True, recents, user

def register_authenticate_and_get_recent_paths(db: Session, email: EmailStr, password: str, username: str):
    user = db.query(User).filter(User.email == email).first()

    if not user:
        new_user = User(email=str(email), password=password, username=username)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        user = new_user
        print(user.id, user.username, user.email)

        return True, [], user
    else:
        return False, [], -1



def get_recent_paths(user_id: int, db: Session):
    last_images = db.query(Image) \
        .filter(Image.user_id == user_id) \
        .order_by(desc(Image.upload_date)) \
        .all()

    paths = [img.path for img in last_images]
    print(paths)

    recents = []
    for path in paths:
        try:
            recents.append(pick_and_convert_base64_image(str(path)))
        except Exception as e:
            print(f"error al convertir la imagen {str(path)} a base64: {e}")

    return recents

def pick_and_convert_base64_image(image_path: str):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string
