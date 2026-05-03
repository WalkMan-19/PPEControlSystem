from sqlalchemy import Column, Integer, String
from app.db.session import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    name = Column(String(25), unique=False, nullable=False)
    surname = Column(String(25), unique=False, nullable=False)
    password = Column(String(128), nullable=False)
