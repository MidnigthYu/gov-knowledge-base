from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from config.settings import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """API Key 鉴权依赖，配置为空时自动放行"""
    if not settings.API_KEY:
        return
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的API Key"
        )