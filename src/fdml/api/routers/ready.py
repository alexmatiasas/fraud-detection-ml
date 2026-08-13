from fastapi import APIRouter

ready_router = APIRouter(prefix="/ready", tags=["ready"])


@ready_router.get("/")
def ready():
    """readiness check (is the model loaded?)

    Args:
        request (Request): _description_

    Returns:
        _type_: _description_
    """
    return 0
