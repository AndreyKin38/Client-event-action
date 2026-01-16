import pandas as pd

from dataclasses import dataclass
from service.schemas import ClientForm


@dataclass
class ClientService:

    @staticmethod
    def get_client(client_id: str) -> ClientForm | None:

        df_full = pd.read_parquet("./preprocessed_data/df_full.parquet")
        client_data = df_full.loc[df_full['client_id'] == client_id]

        if client_data.shape[0]:
            return ClientForm(**client_data.to_dict('records')[0])

        return



