from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, ForeignKey, JSON, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from datetime import datetime, timezone
import uuid
from config import settings

# Database engine and session
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=20,
    max_overflow=0,
    pool_pre_ping=True,
    pool_recycle=300
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=True)  # Nullable for OAuth users
    provider = Column(String(50), nullable=False, default="email")  # email, google, facebook
    is_verified = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    profile_picture = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), 
                       onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Indexes for better performance
    __table_args__ = (
        Index('idx_user_email_provider', 'email', 'provider'),
        Index('idx_user_created_at', 'created_at'),
    )

class RefreshToken(Base):
    __tablename__ = "refresh_tokens"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(255), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    is_revoked = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    __table_args__ = (
        Index('idx_refresh_token_user_expires', 'user_id', 'expires_at'),
    )

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_state = Column(String(50), nullable=False, default="no_product_selected")
    conversation_history = Column(JSONB, nullable=False, default=list)
    product_context = Column(JSONB, nullable=True)
    design_urls = Column(JSONB, nullable=True)  # {design_url, upscaled_url}
    session_metadata = Column(JSONB, nullable=True)  # Additional session data
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), 
                       onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    __table_args__ = (
        Index('idx_user_session_user_updated', 'user_id', 'updated_at'),
    )

class Product(Base):
    __tablename__ = "products"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    external_id = Column(String(100), nullable=False, index=True)  # Printful product ID
    name = Column(String(255), nullable=False)
    brand = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True, index=True)
    subcategory = Column(String(100), nullable=True, index=True)
    specifications = Column(JSONB, nullable=False)  # sizes, colors, techniques, etc.
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), 
                       onupdate=lambda: datetime.now(timezone.utc), nullable=False)

class Order(Base):
    __tablename__ = "orders"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("user_sessions.id", ondelete="SET NULL"), nullable=True)
    external_order_id = Column(String(100), nullable=True, index=True)  # Printful order ID
    order_number = Column(String(50), nullable=False, unique=True, index=True)
    product_data = Column(JSONB, nullable=False)  # Product details at time of order
    design_data = Column(JSONB, nullable=False)  # Design URLs and metadata
    user_selection = Column(JSONB, nullable=False)  # Size, color, placement selection
    status = Column(String(50), nullable=False, default="pending", index=True)
    total_amount = Column(String(20), nullable=True)  # Store as string to avoid decimal issues
    payment_status = Column(String(50), nullable=False, default="pending")
    tracking_info = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), 
                       onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    __table_args__ = (
        Index('idx_order_user_created', 'user_id', 'created_at'),
        Index('idx_order_status_created', 'status', 'created_at'),
    )

# Database dependency for FastAPI
async def get_db() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# Database initialization
async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def close_db():
    """Close database connection"""
    await engine.dispose()