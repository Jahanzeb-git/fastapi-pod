from fastapi import APIRouter, Depends, HTTPException, status, Response, Request, Cookie
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List, Dict, Any
from uuid import UUID
import logging

from auth.dependencies import get_current_user

from app.database import get_db, User
from auth.utils import (
    TokenManager, GoogleOAuthValidator, AuthError,
    validate_email, validate_password
)
from app.config import settings

logger = logging.getLogger(__name__)

# Pydantic Models
class UserResponse(BaseModel):
    id: UUID
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    provider: str
    is_verified: bool
    profile_picture: Optional[str] = None

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    expires_in: int
    user: UserResponse

class SignupRequest(BaseModel):
    email: str
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None

class LoginRequest(BaseModel):
    email: str
    password: str

class GoogleAuthRequest(BaseModel):
    auth_code: str

class RefreshResponse(BaseModel):
    access_token: str
    expires_in: int

class MessageResponse(BaseModel):
    message: str

# Router
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])

@auth_router.post("/email/signup", response_model=TokenResponse)
async def signup(
    request: SignupRequest,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    """Manual email signup with immediate login"""
    try:
        email = validate_email(request.email)
        password = validate_password(request.password)
        
        existing_user_query = select(User).where(User.email == email)
        existing_user = await db.execute(existing_user_query)
        if existing_user.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        password_hash = TokenManager.hash_password(password)
        
        new_user = User(
            email=email,
            password_hash=password_hash,
            provider="email",
            is_verified=True,  # Skip email verification for now
            first_name=request.first_name.strip() if request.first_name else None,
            last_name=request.last_name.strip() if request.last_name else None
        )
        
        db.add(new_user)
        await db.flush()
        await db.refresh(new_user)
        
        access_token = TokenManager.generate_access_token(
            user_id=new_user.id,
            email=new_user.email,
            additional_claims={"provider": new_user.provider}
        )
        
        refresh_token = TokenManager.generate_refresh_token()
        await TokenManager.store_refresh_token(db, new_user.id, refresh_token)
        
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=settings.COOKIE_HTTPONLY,
            secure=settings.COOKIE_SECURE,
            samesite=settings.COOKIE_SAMESITE,
            max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
        
        await db.commit()
        
        logger.info(f"User signed up successfully: {email}")
        
        return TokenResponse(
            access_token=access_token,
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse.from_orm(new_user)
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Signup error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during signup"
        )

@auth_router.post("/email/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    """Manual email login"""
    try:
        email = validate_email(request.email)
        
        user_query = select(User).where(
            User.email == email,
            User.provider == "email",
            User.is_active == True
        )
        user_result = await db.execute(user_query)
        user = user_result.scalar_one_or_none()
        
        if not user or not user.password_hash:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        if not TokenManager.verify_password(request.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        if not user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Please verify your email before logging in"
            )
        
        access_token = TokenManager.generate_access_token(
            user_id=user.id,
            email=user.email,
            additional_claims={"provider": user.provider}
        )
        
        refresh_token = TokenManager.generate_refresh_token()
        await TokenManager.store_refresh_token(db, user.id, refresh_token)
        
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=settings.COOKIE_HTTPONLY,
            secure=settings.COOKIE_SECURE,
            samesite=settings.COOKIE_SAMESITE,
            max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
        
        await db.commit()
        
        logger.info(f"User logged in successfully: {email}")
        
        return TokenResponse(
            access_token=access_token,
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse.from_orm(user)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during login"
        )

@auth_router.post("/google", response_model=TokenResponse)
async def google_auth(
    request: GoogleAuthRequest,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    """Google OAuth authentication"""
    try:
        google_user_info = await GoogleOAuthValidator.verify_google_token(request.auth_code)
        if not google_user_info:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Google authentication code. Please try again."
            )
        
        user_info = GoogleOAuthValidator.validate_google_user_info(google_user_info)
        
        user_query = select(User).where(
            User.email == user_info["email"],
            User.provider == "google"
        )
        user_result = await db.execute(user_query)
        user = user_result.scalar_one_or_none()
        
        if not user:
            user = User(
                email=user_info["email"],
                provider="google",
                is_verified=user_info["is_verified"],
                first_name=user_info["first_name"],
                last_name=user_info["last_name"],
                profile_picture=user_info.get("profile_picture")
            )
            db.add(user)
            await db.flush()
            await db.refresh(user)
            logger.info(f"Created new Google user: {user_info['email']}")
        else:
            user.first_name = user_info["first_name"] or user.first_name
            user.last_name = user_info["last_name"] or user.last_name
            user.profile_picture = user_info.get("profile_picture") or user.profile_picture
            user.is_verified = True
            logger.info(f"Updated existing Google user: {user_info['email']}")
        
        access_token = TokenManager.generate_access_token(
            user_id=user.id,
            email=user.email,
            additional_claims={"provider": user.provider}
        )
        
        refresh_token = TokenManager.generate_refresh_token()
        await TokenManager.store_refresh_token(db, user.id, refresh_token)
        
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=settings.COOKIE_HTTPONLY,
            secure=settings.COOKIE_SECURE,
            samesite=settings.COOKIE_SAMESITE,
            max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
        
        await db.commit()
        
        return TokenResponse(
            access_token=access_token,
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=UserResponse.from_orm(user)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google auth error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during Google authentication"
        )

@auth_router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(
    request: Request,
    response: Response,
    refresh_token: Optional[str] = Cookie(None),
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token using refresh token"""
    try:
        if not refresh_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token not found"
            )
        
        token_record = await TokenManager.verify_refresh_token(db, refresh_token)
        if not token_record:
            response.delete_cookie(
                key="refresh_token",
                httponly=settings.COOKIE_HTTPONLY,
                secure=settings.COOKIE_SECURE,
                samesite=settings.COOKIE_SAMESITE
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        user_query = select(User).where(
            User.id == token_record.user_id,
            User.is_active == True
        )
        user_result = await db.execute(user_query)
        user = user_result.scalar_one_or_none()
        
        if not user:
            await TokenManager.revoke_refresh_token(db, token_record.user_id)
            response.delete_cookie(
                key="refresh_token",
                httponly=settings.COOKIE_HTTPONLY,
                secure=settings.COOKIE_SECURE,
                samesite=settings.COOKIE_SAMESITE
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        new_access_token = TokenManager.generate_access_token(
            user_id=user.id,
            email=user.email,
            additional_claims={"provider": user.provider}
        )
        
        new_refresh_token = TokenManager.generate_refresh_token()
        await TokenManager.store_refresh_token(db, user.id, new_refresh_token)
        
        response.set_cookie(
            key="refresh_token",
            value=new_refresh_token,
            httponly=settings.COOKIE_HTTPONLY,
            secure=settings.COOKIE_SECURE,
            samesite=settings.COOKIE_SAMESITE,
            max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
        
        await db.commit()
        
        logger.info(f"Token refreshed for user: {user.email}")
        
        return RefreshResponse(
            access_token=new_access_token,
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during token refresh"
        )

@auth_router.post("/logout", response_model=MessageResponse)
async def logout(
    response: Response,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Logout user and revoke refresh token"""
    try:
        await TokenManager.revoke_refresh_token(db, current_user.id)
        
        response.delete_cookie(
            key="refresh_token",
            httponly=settings.COOKIE_HTTPONLY,
            secure=settings.COOKIE_SECURE,
            samesite=settings.COOKIE_SAMESITE
        )
        
        logger.info(f"User logged out: {current_user.email}")
        
        return MessageResponse(message="Successfully logged out")
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during logout"
        )
