import fastapi
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from fastapi import status
from .model import predict, instantiate_model
from .payloads import Flights

app = fastapi.FastAPI()
instantiate_model()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=status.HTTP_400_BAD_REQUEST)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(flights: Flights) -> dict:
    return {"predict": predict(flights.to_dataframe())}
