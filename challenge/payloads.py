from typing import Literal
import pandas as pd
from pydantic import BaseModel, conint


class Flight(BaseModel):
    TIPOVUELO: Literal["I", "N"]
    MES: conint(gt=0, le=12)
    OPERA: Literal[
        "K.L.M.",
        "Avianca",
        "Air France",
        "Iberia",
        "American Airlines",
        "British Airways",
        "Air Canada",
        "JetSmart SPA",
        "Gol Trans",
        "Lacsa",
        "Aerolineas Argentinas",
        "Delta Air",
        "United Airlines",
        "Grupo LATAM",
        "Oceanair Linhas Aereas",
        "Aeromexico",
        "Sky Airline",
        "Latin American Wings",
        "Qantas Airways",
        "Alitalia",
        "Austral",
        "Plus Ultra Lineas Aereas",
        "Copa Air",
    ]


class Flights(BaseModel):
    flights: list[Flight]

    def to_dataframe(self):
        return pd.DataFrame([f.dict() for f in self.flights])
