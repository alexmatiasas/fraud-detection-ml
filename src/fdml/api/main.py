from fastapi import FastAPI

from fdml.api.metadata import DESCRIPTION, SUMMARY, TITLE, VERSION, tags_metadata
from fdml.api.routers.health import health_router
from fdml.api.routers.metrics import metrics_router
from fdml.api.routers.model import model_router
from fdml.api.routers.predict import predict_router
from fdml.api.routers.ready import ready_router

app = FastAPI(
    title=TITLE,
    summary=SUMMARY,
    description=DESCRIPTION,
    version=VERSION,
    openapi_url="/openapi.json",
    openapi_tags=tags_metadata,
)

app.include_router(predict_router)
app.include_router(health_router)
app.include_router(ready_router)
app.include_router(model_router)
app.include_router(metrics_router)


@app.get("/")
async def read_root():
    return {"Hello": "World"}
