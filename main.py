import dill
import json
import pandas as pd

from fastapi import FastAPI
from fastapi_app import router


app = FastAPI()
app.include_router(router=router)

