from pydantic import BaseModel, EmailStr, Field, field_validator


def _reject_password_over_bcrypt_byte_limit(value: str) -> str:
    # bcrypt (via passlib's CryptContext, app/core/security.py) silently
    # truncates at 72 BYTES -- everything past that is ignored during both
    # hashing and verification, so two different passwords sharing the same
    # first 72 bytes hash identically and either one logs in. Confirmed
    # directly: hash_password("a"*72 + "tail1") verifies True against
    # "a"*72 + "tail2". This UI is Russian, and Cyrillic is 2 bytes/char in
    # UTF-8, so a Cyrillic passphrase hits this well under 72 *characters* --
    # the check must be on encoded byte length, not len(value).
    if len(value.encode("utf-8")) > 72:
        raise ValueError("Пароль не должен превышать 72 байта")
    return value


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    # The registration form's own `minLength={8}` is client-side only and
    # trivially bypassed by calling this endpoint directly -- confirmed by
    # registering a real account with password="a" before this constraint
    # existed. The frontend hint is UX, not enforcement; this is the actual
    # boundary.
    password: str = Field(min_length=8)

    _validate_password = field_validator("password")(_reject_password_over_bcrypt_byte_limit)


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


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)

    _validate_new_password = field_validator("new_password")(_reject_password_over_bcrypt_byte_limit)
