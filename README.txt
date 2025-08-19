# Industrial Air Pollution Prediction

This project predicts industrial air pollution using LSTM models and air filter data. It includes data preprocessing, model training, evaluation, and exploratory analysis.

## Project Structure

```
data/
    air_filter_data.csv
models/
    fine_tuned_lstm.h5
    pretrained_lstm.h5
notebooks/
    data_exploration.ipynb
src/
    preprocess.py
    train.py
    evaluate.py
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/TShreya-27/industry_air_pollution.git
   cd industry_air_pollution
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- **Preprocessing:**  
  Run `src/preprocess.py` to clean and prepare the data.

- **Training:**  
  Run `src/train.py` to train the LSTM model.

- **Evaluation:**  
  Run `src/evaluate.py` to evaluate model performance.

- **Exploration:**  
  Use `notebooks/data_exploration.ipynb` for data analysis and visualization.

## Models

- `models/pretrained_lstm.h5`: Pretrained LSTM model.
- `models/fine_tuned_lstm.h5`: Fine-tuned LSTM model.

## Data

- `data/air_filter_data.csv`: Raw air filter data for training and evaluation.

## License

This project is licensed under the MIT License.
