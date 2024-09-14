# app/models/admindocuments.py
from pydantic import BaseModel, constr
from typing import Optional

class Standard_Document(BaseModel):
    nameofdoc: str
    description: str 
    index_name: str
    namespace_name: str
    by_admin:bool
