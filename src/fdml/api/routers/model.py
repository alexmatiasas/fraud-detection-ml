from fastapi import APIRouter

model_router = APIRouter(prefix="/model", tags=["models"])


@model_router.get("/info")
def info(req) -> None:
    """current version, run_id, model metrics

    Args:
        req (_type_): _description_
    """


@model_router.put("/reload")
def reload() -> None:
    """(protected) reloads the more recent production model"""
