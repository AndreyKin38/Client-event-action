from pydantic import BaseModel


class ClientForm(BaseModel):
    client_id: str
    session_id: str
    device_category: str | None
    device_os: str | None
    device_brand: str | None
    device_screen_resolution: str | None
    device_browser: str | None
    geo_country: str | None
    geo_city: str | None
    utm_source: str | None
    utm_medium: str | None


class Prediction(BaseModel):
    client_id: str
    result: int

