import torch
from transformers import AutoProcessor, AutoModelForCausalLM


from .schema import Task, CaptionFormat, TaskOptions
from .exceptions import ValidationError


class Model:
  model_id: str
  SUPPORTED_TASKS: set[Task]


class Florence2(Model):

  TASK_TO_TASK_TOKEN = {

    Task.detection: '<OD>',

    # if true will use prompt to do open vocab detection
    (Task.detection, True): '<OPEN_VOCABULARY_DETECTION>',
    (Task.detection, False): '<OD>',

    Task.ocr: '<OCR>',
    (Task.ocr, False): '<OCR>',
    (Task.ocr, True): '<OCR_WITH_REGION>',

    Task.caption: '<CAPTION>',
    (Task.caption, CaptionFormat.short): '<CAPTION>',
    (Task.caption, CaptionFormat.medium): '<DETAILED_CAPTION>',
    (Task.caption, CaptionFormat.long): '<MORE_DETAILED_CAPTION>',
    (Task.caption, CaptionFormat.dense): '<DENSE_REGION_CAPTION>',
  }

  def get_token_for_task(self, task, prompt=None, options: TaskOptions=None):
    if options:
      if task == Task.caption:
        return self.TASK_TO_TASK_TOKEN[task, options.format]
      if task == Task.ocr:
        return self.TASK_TO_TASK_TOKEN[task, options.polys]
    if prompt and task == Task.detection:
      if task == Task.detection:
        return self.TASK_TO_TASK_TOKEN[task, True]
    return self.TASK_TO_TASK_TOKEN[task]

  TASK_TOKEN_TO_TASK = {
    '<CAPTION>': Task.caption,
    '<OD>': Task.detection,
    '<OCR>': Task.ocr,
    '<OCR_WITH_REGION>': Task.ocr,
    '<DETAILED_CAPTION>': Task.caption,
    '<MORE_DETAILED_CAPTION>': Task.caption,
    '<DENSE_REGION_CAPTION>': Task.caption,
    # '<REGION_PROPOSAL>': 'Locate the region proposals in the image.',
    #
    #
    # '<CAPTION_TO_PHRASE_GROUNDING>': "Locate the phrases in the caption: {input}",
    # '<REFERRING_EXPRESSION_SEGMENTATION>': 'Locate {input} in the image with mask',
    # '<REGION_TO_SEGMENTATION>': 'What is the polygon mask of region {input}',
    '<OPEN_VOCABULARY_DETECTION>': Task.detection,  # 'Locate {input} in the image.'
    # '<REGION_TO_CATEGORY>': 'What is the region {input}?',
    # '<REGION_TO_DESCRIPTION>': 'What does the region {input} describe?',
    # '<REGION_TO_OCR>': 'What text is in the region {input}?',
  }



  SUPPORTED_TASKS = set(TASK_TO_TASK_TOKEN.keys())

  # supported tokens are from:
  # https://huggingface.co/microsoft/Florence-2-base/blob/main/processing_florence2.py#L94-L131

  task_prompts_without_inputs = {
    '<OCR>': 'What is the text in the image?',
    '<OCR_WITH_REGION>': 'What is the text in the image, with regions?',
    '<CAPTION>': 'What does the image describe?',
    '<DETAILED_CAPTION>': 'Describe in detail what is shown in the image.',
    '<MORE_DETAILED_CAPTION>': 'Describe with a paragraph what is shown in the image.',
    '<OD>': 'Locate the objects with category name in the image.',
    '<DENSE_REGION_CAPTION>': 'Locate the objects in the image, with their descriptions.',
    '<REGION_PROPOSAL>': 'Locate the region proposals in the image.'
  }
  task_prompts_with_input = {
    '<CAPTION_TO_PHRASE_GROUNDING>': "Locate the phrases in the caption: {input}",
    '<REFERRING_EXPRESSION_SEGMENTATION>': 'Locate {input} in the image with mask',
    '<REGION_TO_SEGMENTATION>': 'What is the polygon mask of region {input}',
    '<OPEN_VOCABULARY_DETECTION>': 'Locate {input} in the image.',
    '<REGION_TO_CATEGORY>': 'What is the region {input}?',
    '<REGION_TO_DESCRIPTION>': 'What does the region {input} describe?',
    '<REGION_TO_OCR>': 'What text is in the region {input}?',
  }



  def __init__(self, model_id, preprocessor_id=None, device=None, dtype=None):
    self.model_id = model_id
    self.preprocessor_id = preprocessor_id or self.model_id
    self.device = device or 'cpu'
    self.dtype = dtype or torch.float32

    self.model = AutoModelForCausalLM.from_pretrained(
      self.model_id,
      torch_dtype=self.dtype,
      trust_remote_code=True
    ).to(device)
    self.processor = AutoProcessor.from_pretrained(
      self.preprocessor_id,
      trust_remote_code=True
    )

  def validate_inputs(self, image, task=None, prompt=None, options=None):
    if task is not None:
      if task not in self.SUPPORTED_TASKS:
        raise ValidationError(f"Task {task} is not supported, should be one of {self.SUPPORTED_TASKS}")

      task_token = self.get_token_for_task(task, prompt=prompt, options=options)

      if prompt and task_token in self.task_prompts_without_inputs:
        raise ValidationError(f"The {task} task does not support including a text prompt")


  def preprocess(self, image, prompt=None, task=None, task_token=None, options=None):
    task_token = task_token or self.get_token_for_task(task, prompt=prompt, options=options)

    formatted_prompt = task_token + (prompt or '')

    return self.processor(
      text=formatted_prompt,
      images=image,
      return_tensors="pt",
    )

  def infer(self, inputs):
    inputs = inputs.to(self.device, dtype=self.dtype)

    outputs = self.model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=2048,
      do_sample=False,
      num_beams=3,
    )

    return outputs

  def postprocess(self, outputs, task, image_size, prompt=None, task_token=None, options=None):
    text = self.processor.batch_decode(outputs, skip_special_tokens=False)[0]

    task_token = task_token or self.get_token_for_task(task, prompt=prompt, options=options)

    parsed_results = self.processor.post_process_generation(
      text,
      task=task_token,
      image_size=image_size,
    )

    parsed_results = {
      self.TASK_TOKEN_TO_TASK[token]: data for token, data in parsed_results.items()
    }

    return text, parsed_results


  def __call__(self, image, task: Task, prompt: str = None, options=None, task_token=None):
    self.validate_inputs(image=image, task=task, prompt=prompt, options=options)

    inputs = self.preprocess(
      image,
      prompt=prompt,
      task=task,
      task_token=task_token,
      options=options
    )

    outputs = self.infer(inputs)

    text_output, parsed_results = self.postprocess(
      outputs,
      task=task,
      prompt=prompt,
      task_token=task_token,
      image_size=(image.width, image.height),
      options=options
    )

    return text_output, parsed_results

