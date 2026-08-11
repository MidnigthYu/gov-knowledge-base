from pydantic import BaseModel

class ApiResponse(BaseModel):
    code: int = 0
    message: str = "success"
    data: dict = None