from fastapi import APIRouter, Request

metrics_router = APIRouter(prefix="/metrics", tags=["metrics"])


@metrics_router.get("/")
def info(request: Request):
    """Prometheus-style, if you add observability

    Args:
        request (Request): _description_

    Returns:
        _type_: _description_
    """
    return 0
