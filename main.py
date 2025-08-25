from src.models import HybridNeuromorphicModel
from src.data import PlasmaDataLoader

# Load model
model = HybridNeuromorphicModel()

# Load data
data_loader = PlasmaDataLoader("data/processed/")

# Train
model.train(data_loader)

# Detect anomalies
anomalies = model.detect_anomalies(plasma_sequence)