from app.database import get_db, User
from auth.utils import TokenManager, AuthError

# Security scheme
security = HTTPBearer()

# Dependency for getting current user
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Dependency to get current authenticated user"""
    try:
        # Extract and decode token
        token = credentials.credentials
        payload = TokenManager.decode_access_token(token)
        
        user_id = UUID(payload.get("user_id"))
        email = payload.get("email")
        
        if not user_id or not email:
            raise AuthError("Token missing required claims")
        
        # Get user from database
        user_query = select(User).where(
            User.id == user_id,
            User.email == email,
            User.is_active == True
        )
        user_result = await db.execute(user_query)
        user = user_result.scalar_one_or_none()
        
        if not user:
            raise AuthError("User not found")
        
        return user
        
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID format",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )

# Optional dependency for getting current user (returns None if not authenticated)
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Optional dependency to get current user (returns None if not authenticated)"""
    if not credentials:
        return None
    
    try:
        # Use the main get_current_user logic but don't raise exceptions
        token = credentials.credentials
        payload = TokenManager.decode_access_token(token)
        
        user_id = UUID(payload.get("user_id"))
        user_query = select(User).where(
            User.id == user_id,
            User.is_active == True
        )
        user_result = await db.execute(user_query)
        return user_result.scalar_one_or_none()
        
    except Exception:
        return None400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Signup error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during signup"
        )