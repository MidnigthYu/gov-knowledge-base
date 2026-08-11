from fastapi import APIRouter, Depends
from app.schemas.request import BuildKnowledgeReq, AddDocumentReq
from common.auth import verify_api_key

router = APIRouter(
    prefix="/api/knowledge",
    tags=["知识库管理"],
    dependencies=[Depends(verify_api_key)]
)

@router.post("/build", summary="批量构建知识库")
def build_knowledge(req: BuildKnowledgeReq):
    from app.deps import kb_manager
    from app.schemas.response import ApiResponse
    result = kb_manager.build_from_dir(req.docs_dir)
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