import fastapi
import uuid

from fastapi import HTTPException

from .models import Florence2
from .schema import VisionRequest, VisionResponse, convert_results, ImageMetadata
from .image import base64_decode
from . import config

app = fastapi.FastAPI()

models = {
  config.DEFAULT_MODEL: Florence2(model_id=config.DEFAULT_MODEL)
}


@app.post('/vision')
def vision(body: VisionRequest) -> VisionResponse:
  if body.model and body.model not in models:
    raise HTTPException(
      status_code=400,
      detail=f'Unsupported model provided {body.model}, should be one of {", ".join(models)}'
    )

  model = models[body.model or config.DEFAULT_MODEL]

  image = base64_decode(body.image)

  text, model_results = model(
    image=image,
    task=body.task,
    prompt=body.prompt,
    options=body.options
  )

  results = convert_results(model_results)

  return VisionResponse(
    id=body.id or str(uuid.uuid4()),
    model=model.model_id,
    results=results,
    image=ImageMetadata(
      height=image.height,
      width=image.width,
    )
  )


@app.get("/")
async def index():
  urls = [{"path": route.path, "method": route.methods} for route in app.routes]
  return {"urls": urls}



