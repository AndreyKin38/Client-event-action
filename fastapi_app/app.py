import pandas as pd

from fastapi import APIRouter
from model import event_action_pipeline
from service import Prediction, ClientForm, ClientService


router = APIRouter(prefix="/model", tags=["model"])


@router.get('/status')
def status():
    res = event_action_pipeline()['metadata']['name']
    return res


@router.post(
    '/predict',
    response_model=Prediction | str
)
def predict(
        client_id: str,
        client_service: ClientService
):

    form = client_service.get_client(client_id)

    if isinstance(form, ClientForm):
        data = pd.DataFrame.from_dict([form.dict()])
        model = event_action_pipeline()['model']
        result = model.predict(data)

        return Prediction(
            client_id=client_id,
            result=result
        )
    else:
        return f'The client_id: {client_id} was not found'


