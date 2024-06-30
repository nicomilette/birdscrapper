
# birdscrapper 

## Description

These scripts are designed to scrape bird call recordings, generate spectrograms, and train an AI model to classify bird calls based on these recordings. The scripts are used for different stages of data processing, analysis, and model training.

## Features

- **Data Scraping**: Fetch bird call recordings from the xeno-canto API.
- **Data Cleaning**: Filter and clean the dataset for training.
- **Spectrogram Generation**: Convert audio recordings into spectrogram images.
- **Data Analysis**: Generate and display statistics about the bird recordings.
- **Model Training**: Train a convolutional neural network (CNN) to classify bird calls based on spectrogram images.

## Requirements

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/birdscrapper.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Global configuration variables are stored in `src/config.py`. This file includes settings for file paths, thresholds, and other parameters used across different scripts.

## Usage

Run the main script to access all functionalities via a menu:
```bash
python src/main.py
```

## Statistics and Analysis

The project includes several statistical analyses and visualizations, such as:
- Number of recordings per bird species
- Average number of recordings
- Species with maximum and minimum geographical coordinates

## Data Licensing

The data scraped by these scripts is licensed under the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). For more information, visit [Creative Commons License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
