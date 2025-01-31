import os
from src.data.load_data import load_data
from src.data.preprocess import preprocess
from src.features.build_features import build_features
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from src.visualization.visualize import visualize

def main():
    # Load the data
    raw_data_path = os.path.join('data', 'raw')
    data = load_data(raw_data_path)
    
    # Preprocess the data
    processed_data = preprocess(data)
    
    # Build features
    features = build_features(processed_data)
    
    # Train the model
    model = train_model(features)
    
    # Evaluate the model
    evaluation = evaluate_model(model, features)
    
    # Visualize the results
    visualize(evaluation)

if __name__ == "__main__":
    main()
