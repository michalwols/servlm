import requests
from PIL import Image
from typing import Optional

from .image import base64_encode


class SerVLMException(Exception):
  response: requests.Response

  def __init__(self, response: requests.Response):
    self.response = response


class SerVLMClient:
  def __init__(self, base_url='http://127.0.0.1:8000'):
    self.base_url = base_url

  def vision(
      self,
      image: str | Image.Image,
      prompt: str = None,
      task: str = None,
      id: str = None,
      model: str = None,
      options: Optional[dict] = None
  ):
    response = requests.post(f'{self.base_url}/vision', json=dict(
      id=id,
      image=base64_encode(image, url_prefix=True) if isinstance(image, Image.Image) else image,
      prompt=prompt,
      task=task,
      model=model,
      options=options
    ))

    if response.ok:
      return response.json()

    else:
      raise SerVLMException(response)

  def ocr(
      self,
      image: str | Image.Image,
      polys=False,
      id: str = None,
      model: str = None,
  ):
    return self.vision(
      image=image,
      id=id,
      model=model,
      task='ocr',
      options=dict(polys=polys)
    )

  def caption(
      self,
      image: str | Image.Image,
      prompt: str = None,
      id: str = None,
      model: str = None,
      format='short'
  ):
    return self.vision(
      image=image,
      prompt=prompt,
      id=id,
      model=model,
      task='caption',
      options=dict(format=format)
    )

  def detect(
      self,
      image: str | Image.Image,
      prompt: str = None,
      id: str = None,
      model: str = None, ):
    return self.vision(
      image=image,
      prompt=prompt,
      id=id,
      model=model,
      task='detection'
    )
