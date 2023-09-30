## Choosing a model
The model chosen was `6.b.i. XGBoost with Feature Importance and with Balance`. I did not find any significant difference between `6.b.i` and `6.b.iii`, both models have similar accuracy. I decided to go with `6.b.i` because it had fewer incorrect results, although the number is not significant at all.

## Modifications to test_model
The predict test had to be modified because it attempted to make a prediction before the model had been trained. Additional code was added to train the model before running `.predict()`. An argument could be made that training is expensive and therefore it is better to share the same model across tests. However, sharing state across tests is considered an anti-pattern, the test result might depend on the execution order. An alternate approach would be to run a single integration test for the model that calls `preprocess`, `fit` and `predict`.

## API and general design

### Model
- The real model is behind `DelayModel`. Additionally, `DelayModel` is behind a module singleton `ModelContainer` and the classifier model is loaded once from the saved parameters in `xgb_model.json` as soon as the API starts.
- Instead of exposing `DelayModel` or `ModelContainer` a single `predict()` function is exposed at the module level to simplify the module API.-
- A `DelayModel.preprocess_and_predict()` method is exposed to simplify the usage of `DelayModel`, otherwise the user would have to remember to always call `preprocess()` before calling `predict()`

### API
- Although `FastAPI` default response when data fails validation is `HTTP_422_UNPROCESSABLE_ENTITY`, the tests expected an `HTTP_400_BAD_REQUEST`. An extra function to handle `RequestValidationError` was added to return the expected code.
- The expected payload format is represented in `payloads.py`. `Literal` was used instead of `Enum` for simplicity and time constraint, the latter is preferred.
- Only the `"OPERA"` values from `data.csv` are listed as valid choices. The list was manually constructed, in further improvements and iterations the values could be read from configuration files.
- A basic `to_dataframe()` method was added to `Flights` to avoid passing `pydantic` objects to `model` as a good practice. The `model` module should only receive `DataFrame`, any conversion should be made by the caller, in this case `Flights`.

## Hosting, CI/CD
- The app is hosted on [Render](https://render.com/), the provider was chosen because it has direct support for FastAPI.
- The IaC file `render.yaml` is straightforward and much simpler than other IaC models such as Terraform.
- `Render` also has a free tier that provides enough resources to run the model.
- **NOTE**: While the free tier can run the model with an acceptable response time, the CPU provided for training is low and therefore it may take up to 5 minutes from deployment to readiness status.
- The CI flow runs both `model-test` and `api-test`
- The CD flow consists of a single call to [Render Deploy Hook](https://render.com/docs/deploy-hooks). The hook URL is stored as a secret in this repository.