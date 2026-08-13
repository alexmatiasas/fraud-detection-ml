from fastapi import APIRouter, status

health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.get("/", status_code=status.HTTP_200_OK)
def health():
    """checkness of liveness (is the api living during process?)

    Returns:
        _type_: _description_
    """
    return "ラーメン"
