import mlflow
logged_model = 'file:///mnt/d/Onderive/Particular/onedrive/Projetos/python/mlflow2/mlflow2/mlruns/1/d9b2e668d7754a9a93c8342f28df94a8/artifacts/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = pd.read_csv('data/processed/casas_X.csv')

#print(data)
predicted = loaded_model.predict(data)

data['predicted'] = predicted
data.to_csv('data/processed/precos.csv')
