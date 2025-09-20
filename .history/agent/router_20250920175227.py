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
        # Validate and normalize email
        email = validate_email(request.email)
        
        # Validate password
        password = validate_password(request.password)
        
        # Check if user already exists
        existing_user_query = select(User).where(User.email == email)
        existing_user = await db.execute(existing_user_query)
        if existing_user.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Hash password
        password_hash = TokenManager.hash_password(password)
        
        # Create user
        new_user = User(
            email=email,
            password_hash=password_hash,
            provider="email",
            is_verified=True,  # Skip email verification for now
            first_name=request.first_name.strip() if request.first_name else None,
            last_name=request.last_name.strip() if request.last_name else None
        )
        
        db.add(new_user)
        await db.flush()  # Get the user ID
        await db.refresh(new_user)
        
        # Generate tokens
        access_token = TokenManager.generate_access_token(
            user_id=new_user.id,
            email=new_user.email,
            additional_claims={"provider": new_user.provider}
        )
        
        refresh_token = TokenManager.generate_refresh_token()
        await TokenManager.store_refresh_token(db, new_user.id, refresh_token)
        
        # Set refresh token as HttpOnly cookie
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=settings.COOKIE_HTTPONLY,
            secure=settings.COOKIE_SECURE,
            samesite=settings.COOKIE_SAMESITE,
            max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
        
        await db.commit()
        
        # Prepare user data for response
        user_data = {
            "id": str(new_user.id),
            "email": new_user.email,
            "first_name": new_user.first_name,
            "last_name": new_user.last_name,
            "provider": new_user.provider,
            "is_verified": new_user.is_verified
        }
        
        logger.info(f"User signed up successfully: {email}")
        
        return TokenResponse(
            access_token=access_token,
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user_data
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
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
        
        # Verify refresh token
        token_record = await TokenManager.verify_refresh_token(db, refresh_token)
        if not token_record:
            # Clear invalid refresh token cookie
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
        
        # Get user
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
        
        # Generate new access token
        new_access_token = TokenManager.generate_access_token(
            user_id=user.id,
            email=user.email,
            additional_claims={"provider": user.provider}
        )
        
        # Generate new refresh token and store it
        new_refresh_token = TokenManager.generate_refresh_token()
        await TokenManager.store_refresh_token(db, user.id, new_refresh_token)
        
        # Set new refresh token cookie
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
        # Revoke all refresh tokens for the user
        await TokenManager.revoke_refresh_token(db, current_user.id)
        
        # Clear refresh token cookie
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

@auth_router.post("/email/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db)
):
    """Manual email login"""
    try:
        email = validate_email(request.email)
        
        # Find user
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
        
        # Verify password
        if not TokenManager.verify_password(request.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Check if user is verified (for future email verification)
        if not user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Please verify your email before logging in"
            )
        
        # Generate tokens
        access_token = TokenManager.generate_access_token(
            user_id=user.id,
            email=user.email,
            additional_claims={"provider": user.provider}
        )
        
        refresh_token = TokenManager.generate_refresh_token()
        await TokenManager.store_refresh_token(db, user.id, refresh_token)
        
        # Set refresh token as HttpOnly cookie
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=settings.COOKIE_HTTPONLY,
            secure=settings.COOKIE_SECURE,
            samesite=settings.COOKIE_SAMESITE,
            max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
        
        await db.commit()
        
        # Prepare user data for response
        user_data = {
            "id": str(user.id),
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "provider": user.provider,
            "is_verified": user.is_verified
        }
        
        logger.info(f"User logged in successfully: {email}")
        
        return TokenResponse(
            access_token=access_token,
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user_data
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
        # Verify Google auth code and get user info
        google_user_info = await GoogleOAuthValidator.verify_google_token(request.auth_code)
        if not google_user_info:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Google authentication is not fully implemented yet"
            )
        
        # Validate and normalize Google user info
        user_info = GoogleOAuthValidator.validate_google_user_info(google_user_info)
        
        # Check if user exists
        user_query = select(User).where(
            User.email == user_info["email"],
            User.provider == "google"
        )
        user_result = await db.execute(user_query)
        user = user_result.scalar_one_or_none()
        
        if not user:
            # Create new user from Google info
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
            # Update existing user info
            user.first_name = user_info["first_name"] or user.first_name
            user.last_name = user_info["last_name"] or user.last_name
            user.profile_picture = user_info.get("profile_picture") or user.profile_picture
            user.is_verified = True  # Google users are always verified
            logger.info(f"Updated existing Google user: {user_info['email']}")
        
        # Generate tokens
        access_token = TokenManager.generate_access_token(
            user_id=user.id,
            email=user.email,
            additional_claims={"provider": user.provider}
        )
        
        refresh_token = TokenManager.generate_refresh_token()
        await TokenManager.store_refresh_token(db, user.id, refresh_token)
        
        # Set refresh token as HttpOnly cookie
        response.set_cookie(
            key="refresh_token",
            value=refresh_token,
            httponly=settings.COOKIE_HTTPONLY,
            secure=settings.COOKIE_SECURE,
            samesite=settings.COOKIE_SAMESITE,
            max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )
        
        await db.commit()
        
        # Prepare user data for response
        user_data = {
            "id": str(user.id),
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "provider": user.provider,
            "is_verified": user.is_verified,
            "profile_picture": user.profile_picture
        }
        
        return TokenResponse(
            access_token=access_token,
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user_data
        )
        
    except HTTPException:
        raise
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_