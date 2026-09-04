from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.rate_limit import rate_limit
from app.core.security import create_access_token, hash_password, verify_password
from app.db.session import get_db
from app.models.patient import Patient
from app.models.user import User
from app.schemas.user import LoginRequest, PasswordChange, Token, UserCreate, UserRead

router = APIRouter(prefix="/auth", tags=["auth"])

_auth_rate_limit = rate_limit(
    "auth", max_attempts=settings.login_rate_limit_attempts, window_seconds=settings.login_rate_limit_window_seconds
)
# Own budget/prefix, deliberately not shared with _auth_rate_limit: this
# endpoint requires a valid token (an attacker guessing a logged-in user's
# current password can't also be exhausting the login attempt budget for
# that IP, and a legitimate user's own earlier login attempts shouldn't eat
# into their change-password budget or vice versa).
_change_password_rate_limit = rate_limit(
    "change_password",
    max_attempts=settings.login_rate_limit_attempts,
    window_seconds=settings.login_rate_limit_window_seconds,
)


@router.get("/me", response_model=UserRead)
def get_me(current_user: User = Depends(get_current_user)) -> User:
    return current_user


@router.post(
    "/register", response_model=UserRead, status_code=status.HTTP_201_CREATED, dependencies=[Depends(_auth_rate_limit)]
)
def register(payload: UserCreate, db: Session = Depends(get_db)) -> User:
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Пользователь с таким email уже существует")

    user = User(
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.flush()

    # Every account is self-service: each user gets their own patient record
    # under the hood so the existing studies/analysis data model doesn't need
    # to change, but nothing in the product surfaces the word "patient".
    db.add(Patient(full_name=user.full_name, created_by_id=user.id))
    try:
        db.commit()
    except IntegrityError:
        # Narrows a race the check above can't close on its own: two
        # genuinely concurrent registration requests for the same email can
        # both pass the check before either commits (User.email is unique).
        # Without this, the second commit would raise an unhandled
        # IntegrityError -- same class of bug as analysis.py's double-submit
        # fix -- instead of the intended clean 409.
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Пользователь с таким email уже существует")
    db.refresh(user)
    return user


@router.post("/login", response_model=Token, dependencies=[Depends(_auth_rate_limit)])
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> Token:
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный email или пароль",
        )
    return Token(access_token=create_access_token(user.id))


@router.post(
    "/change-password",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(_change_password_rate_limit)],
)
def change_password(
    payload: PasswordChange,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    if not verify_password(payload.current_password, current_user.hashed_password):
        # 400, not 401: frontend/src/lib/api.ts's request() treats ANY 401
        # response that carried a token as "your session expired" and
        # force-clears the token + redirects to /login (correct for every
        # other endpoint here, where 401 can only mean an invalid/expired
        # JWT). This request DOES carry a valid token -- the user is exactly
        # who they say they are, they just mistyped their current password
        # -- so a 401 here would incorrectly log out an authenticated user
        # over a typo. 400 keeps this endpoint out of that codepath.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Неверный текущий пароль")
    # Tokens are stateless JWTs with no session/version tracking anywhere in
    # this codebase (no logout blacklist either) -- changing the password
    # does not invalidate a token already issued elsewhere. Consistent with
    # the rest of the auth model as it stands today; revoking other sessions
    # on password change would need a token-version column plus a check in
    # decode_access_token, a bigger change than this endpoint's own scope.
    current_user.hashed_password = hash_password(payload.new_password)
    db.commit()
