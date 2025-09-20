import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://printplatform_user:secure_password_123@localhost:5432/printplatform_db",
        description="PostgreSQL database URL"
    )
    
    # JWT Configuration
    JWT_SECRET_KEY: str = Field(
        default="your-super-secret-jwt-key-change-in-production",
        description="Secret key for JWT token signing"
    )
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiry in minutes")
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, description="Refresh token expiry in days")
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    
    # Google OAuth Configuration
    GOOGLE_CLIENT_ID: str = Field(default="", description="Google OAuth client ID")
    GOOGLE_CLIENT_SECRET: str = Field(default="", description="Google OAuth client secret")
    GOOGLE_REDIRECT_URI: str = Field(default="http://localhost:8000/auth/google/callback", description="Google OAuth redirect URI")
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")
    DEBUG: bool = Field(default=True, description="Debug mode")
    
    # OpenAI API key Configuration
    OPENAI_API_KEY: str = Field(
        default="76fc8053194685a65fb8d82f723d046e9c99d79a803efbe88a55a2169f2ba63d",
        description="OpenAI's API key for GPT-5"
    )
    OPENAI_MODEL: str = Field(default="gpt-4", description="OpenAI model to use")
    
    # Printful API Configuration
    PRINTFUL_API_BASE: str = Field(default="https://api.printful.com/v2", description="Printful API base URL")
    PRINTFUL_AUTH_TOKEN: str = Field(
        default="Bearer 3nbHw2A3tnA4mTy4q8TOivVJB1ureKe99OBNR2ym",
        description="Printful API authentication token"
    )
    
    # Security Configuration
    BCRYPT_ROUNDS: int = Field(default=12, description="Bcrypt hashing rounds")
    COOKIE_SECURE: bool = Field(default=False, description="Secure cookies (HTTPS only)")
    COOKIE_HTTPONLY: bool = Field(default=True, description="HTTP only cookies")
    COOKIE_SAMESITE: str = Field(default="lax", description="Cookie SameSite policy")
    
    # Email Configuration (placeholder for future implementation)
    SMTP_SERVER: Optional[str] = Field(default=None, description="SMTP server")
    SMTP_PORT: Optional[int] = Field(default=587, description="SMTP port")
    SMTP_USERNAME: Optional[str] = Field(default=None, description="SMTP username")
    SMTP_PASSWORD: Optional[str] = Field(default=None, description="SMTP password")

    # WebSocket Configuration
    WEBSOCKET_PING_INTERVAL: int = Field(default=30, description="WebSocket ping interval in seconds")
    WEBSOCKET_PING_TIMEOUT: int = Field(default=10, description="WebSocket ping timeout in seconds")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()