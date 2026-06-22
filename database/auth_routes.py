from pydantic import BaseModel

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

from fastapi import APIRouter, HTTPException
from database.auth import register_user, login_user
from database.jwt_handler import create_access_token

router = APIRouter()

@router.post("/register")
def register(request: RegisterRequest):

    success = register_user(
        request.username,
        request.password
    )

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )

    return {"message": "User registered successfully"}


@router.post("/login")
def login(request: LoginRequest):

    user = login_user(
        request.username,
        request.password
    )

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    token = create_access_token(
        user["id"],
        user["username"]
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user["id"]
    }