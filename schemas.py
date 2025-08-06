from pydantic import BaseModel
from typing import List, Union

class HackRxRequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]