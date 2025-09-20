import jwt
import bcrypt
import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from app.database import RefreshToken
from app.config import settings

class AuthError(Exception):
    """Custom authentication error"""
    pass

class TokenManager:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt(rounds=settings.BCRYPT_ROUNDS)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    @staticmethod
    def generate_access_token(user_id: str, email: str, additional_claims: Optional[Dict] = None) -> str:
        """Generate JWT access token"""
        now = datetime.now(timezone.utc)
        payload = {
            "user_id": str(user_id),
            "email": email,
            "type": "access",
            "iat": now,
            "exp": now + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
            "iss": "printplatform-api",
            "aud": "printplatform-client"
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    
    @staticmethod
    def generate_refresh_token() -> str:
        """Generate secure random refresh token"""
        return secrets.token_urlsafe(64)
    
    @staticmethod
    def hash_refresh_token(token: str) -> str:
        """Hash refresh token for storage"""
        return hashlib.sha256(token.encode('utf-8')).hexdigest()
    
    @staticmethod
    async def store_refresh_token(db: AsyncSession, user_id: UUID, token: str) -> RefreshToken:
        """Store hashed refresh token in database"""
        # First, revoke any existing refresh tokens for this user
        await db.execute(
            delete(RefreshToken).where(RefreshToken.user_id == user_id)
        )
        
        # Create new refresh token
        refresh_token = RefreshToken(
            user_id=user_id,
            token_hash=TokenManager.hash_refresh_token(token),
            expires_at=datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        )
        
        db.add(refresh_token)
        await db.commit()
        return refresh_token
    
    @staticmethod
    async def verify_refresh_token(db: AsyncSession, token: str) -> Optional[RefreshToken]:
        """Verify refresh token and return token record if valid"""
        token_hash = TokenManager.hash_refresh_token(token)
        
        query = select(RefreshToken).where(
            RefreshToken.token_hash == token_hash,
            RefreshToken.expires_at > datetime.now(timezone.utc),
            RefreshToken.is_revoked == False
        )
        
        result = await db.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def revoke_refresh_token(db: AsyncSession, user_id: UUID) -> None:
        """Revoke all refresh tokens for a user"""
        await db.execute(
            delete(RefreshToken).where(RefreshToken.user_id == user_id)
        )
        await db.commit()
    
    @staticmethod
    def decode_access_token(token: str) -> Dict[str, Any]:
        """Decode and verify access token"""
        try:
            payload = jwt.decode(
                token, 
                settings.JWT_SECRET_KEY, 
                algorithms=[settings.JWT_ALGORITHM],
                audience="printplatform-client",
                issuer="printplatform-api"
            )
            
            # Verify token type
            if payload.get("type") != "access":
                raise AuthError("Invalid token type")
            
            return payload
        
        except jwt.ExpiredSignatureError:
            raise AuthError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthError(f"Invalid token: {str(e)}")
    
    @staticmethod
    def extract_token_from_header(authorization_header: str) -> str:
        """Extract token from Authorization header"""
        if not authorization_header:
            raise AuthError("Authorization header is missing")
        
        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise AuthError("Invalid authorization header format")
        
        return parts[1]

class GoogleOAuthValidator:
    @staticmethod
    async def verify_google_token(auth_code: str) -> Optional[Dict[str, Any]]:
        """
        Verify Google OAuth authorization code and return user info
        This is a placeholder - you'll need to implement the actual Google OAuth flow
        """
        # TODO: Implement actual Google OAuth verification
        # This should:
        # 1. Exchange auth_code for access token
        # 2. Use access token to get user info from Google
        # 3. Return user info dict with email, name, picture, etc.
        
        # For now, return None to indicate not implemented
        return None
    
    @staticmethod
    def validate_google_user_info(user_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize Google user info"""
        if not user_info.get("email"):
            raise AuthError("Google user info missing email")
        
        if not user_info.get("email_verified", False):
            raise AuthError("Google email is not verified")
        
        return {
            "email": user_info["email"].lower().strip(),
            "first_name": user_info.get("given_name", "").strip(),
            "last_name": user_info.get("family_name", "").strip(),
            "profile_picture": user_info.get("picture"),
            "is_verified": True
        }

# Utility functions for validation
def validate_email(email: str) -> str:
    """Basic email validation and normalization"""
    email = email.lower().strip()
    if "@" not in email or len(email) < 5:
        raise ValueError("Invalid email format")
    return email

def validate_password(password: str) -> str:
    """Password validation"""
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long")
    if not any(c.isupper() for c in password):
        raise ValueError("Password must contain at least one uppercase letter")
    if not any(c.islower() for c in password):
        raise ValueError("Password must contain at least one lowercase letter")
    if not any(c.isdigit() for c in password):
        raise ValueError("Password must contain at least one digit")
    return password

def generate_order_number() -> str:
    """Generate unique order number"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_part = secrets.token_hex(4).upper()
    return f"ORD-{timestamp}-{random_part}"