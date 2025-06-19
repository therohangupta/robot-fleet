from pydantic import BaseModel

class TaskRequest(BaseModel):
    task_description: str

class TaskResult(BaseModel):
    success: bool
    message: str
    replan: bool 