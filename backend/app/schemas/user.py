from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    # The registration form's own `minLength={8}` is client-side only and
    # trivially bypassed by calling this endpoint directly -- confirmed by
    # registering a real account with password="a" before this constraint
    # existed. The frontend hint is UX, not enforcement; this is the actual
    # boundary.
    password: str = Field(min_length=8)


class UserRead(BaseModel):
    id: str
    email: EmailStr
    full_name: str
    role: str

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str
