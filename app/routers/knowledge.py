import os
import time
from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, UploadFile
from app.schemas.request import BuildKnowledgeReq, AddDocumentReq
from common.auth import verify_api_key
from config.settings import get_settings

router = APIRouter(
    prefix="/api/knowledge",
    tags=["知识库管理"],
    dependencies=[Depends(verify_api_key)]
)

@router.post("/build", summary="批量构建知识库")
def build_knowledge(req: BuildKnowledgeReq):
    from app.deps import kb_manager
    from app.schemas.response import ApiResponse
    result = kb_manager.build_from_dir(req.docs_dir, collection_name=req.collection_name)
    return ApiResponse(data=result)

@router.get("/list", summary="获取所有知识库列表")
def list_knowledge_bases():
    from app.deps import kb_manager
    from app.schemas.response import ApiResponse
    collections = kb_manager.list_knowledge_bases()
    return ApiResponse(data={"collections": collections})

@router.post("/add-document", summary="单文档增量入库")
def add_single_document(req: AddDocumentReq):
    from app.deps import kb_manager
    from app.schemas.response import ApiResponse
    chunk_count = kb_manager.add_single_document(
        file_path=req.file_path,
        collection_name=req.collection_name
    )
    return ApiResponse(data={"added_chunks": chunk_count})

@router.delete("/{collection_name}", summary="删除指定知识库")
def delete_knowledge_base(collection_name: str):
    from app.deps import kb_manager
    from app.schemas.response import ApiResponse
    kb_manager.delete_knowledge_base(collection_name)
    return ApiResponse(data={"deleted_collection": collection_name})

@router.post("/upload", summary="单文件上传并增量入库")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form(None),
    settings = Depends(get_settings)
):
    from app.deps import kb_manager
    from app.schemas.response import ApiResponse
    from common.exceptions import ErrorCode, GovRAGBaseError

    # 文件格式白名单校验
    file_suffix = Path(file.filename).suffix.lower()
    if file_suffix not in settings.UPLOAD_ALLOWED_SUFFIX:
        raise GovRAGBaseError(ErrorCode.FILE_FORMAT_NOT_SUPPORTED)

    # 文件名安全过滤：剥离路径字符，防范路径遍历漏洞
    safe_filename = os.path.basename(file.filename)

    # 初始化临时目录
    temp_dir = Path(settings.UPLOAD_TEMP_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 生成带时间戳的唯一临时文件名，避免覆盖
    temp_filename = f"{int(time.time())}_{safe_filename}"
    temp_file_path = temp_dir / temp_filename

    try:
        # 读取文件内容
        file_content = await file.read()

        # 文件大小校验
        if len(file_content) > settings.MAX_UPLOAD_SIZE:
            raise GovRAGBaseError(ErrorCode.FILE_SIZE_EXCEED)

        # 空文件拦截
        if not file_content.strip():
            raise GovRAGBaseError(ErrorCode.PARAM_INVALID, detail="上传文件内容为空")

        with open(temp_file_path, "wb") as f:
            f.write(file_content)

        # 复用现有入库逻辑，透传集合名
        target_collection = collection_name or settings.CHROMA_DEFAULT_COLLECTION
        added_chunks = kb_manager.add_single_document(
            file_path=str(temp_file_path),
            collection_name=target_collection
        )

        return ApiResponse(data={
            "added_chunks": added_chunks,
            "collection_name": target_collection,
            "filename": safe_filename
        })

    finally:
        # 兜底清理：无论成功失败都删除临时文件
        if temp_file_path.exists():
            os.remove(temp_file_path)