from pydantic import BaseModel
from typing import Optional, Any, Literal


from enum import StrEnum


class BoundingBox(BaseModel):
  xyxy: list[float]
  label: Optional[str] = None


class Polygon(BaseModel):
  points: list[float]
  label: Optional[str] = None


class Task(StrEnum):
  caption = 'caption'
  detection = 'detection'
  ocr = 'ocr'


class CaptionFormat(StrEnum):
  short = 'short'
  medium = 'medium'
  long = 'long'
  dense = 'dense'

class CaptionOptions(BaseModel):
  format: Optional[CaptionFormat] = CaptionFormat.short

class CaptionResults(BaseModel):
  text: Optional[str] = None
  boxes: Optional[list[BoundingBox]] = None


class OCROptions(BaseModel):
  polys: bool = False

class OCRResults(BaseModel):
  text: Optional[str] = None
  polys: Optional[list[Polygon]] = None



class DetectionResults(BaseModel):
  boxes: list[BoundingBox]


class ImageMetadata(BaseModel):
  width: Optional[int] = None
  height: Optional[int] = None

TaskOptions = CaptionOptions | OCROptions | None


class VisionRequest(BaseModel):
  id: Optional[str] = None
  model: Optional[str] = None

  task: Optional[Task] = None
  options: TaskOptions = None

  image: str
  prompt: Optional[str] = None


class VisionResponse(BaseModel):
  id: str
  model: Optional[str] = None

  results: dict[Task, DetectionResults | CaptionResults | OCRResults] = None

  image: ImageMetadata = None





def convert_results(model_results):
  results = {}
  for task, mr in model_results.items():
    if task == Task.detection:
      labels = 'labels' if 'labels' in mr else 'bboxes_labels'
      results[task] = DetectionResults(boxes=[
        BoundingBox(
          xyxy=xyxy,
          label=label
        ) for xyxy, label in zip(mr['bboxes'], mr[labels])
      ])


    if task == Task.ocr:
      if isinstance(mr, str):
        results[task] = OCRResults(text=mr)
      elif 'quad_boxes' in mr:
        results[task] = OCRResults(
          polys=[
            Polygon(
              points=xyxy,
              label=label
            ) for xyxy, label in zip(mr['quad_boxes'], mr['labels'])
          ]
        )


    if task == Task.caption:
      if 'bboxes' in mr:
        results[task] = CaptionResults(
          boxes=[
            BoundingBox(
              xyxy=xyxy,
              label=label
            ) for xyxy, label in zip(mr['bboxes'], mr['labels'])
          ]
        )
      else:
        results[task] = CaptionResults(text=mr)
  return results