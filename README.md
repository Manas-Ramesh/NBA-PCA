# NBA Player Clustering Using PCA

This project performs dimensionality reduction and clustering of NBA player statistics using Principal Component Analysis (PCA) to identify similar players who can serve as replacements.

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete analysis pipeline:

```python
from nba_pca_analysis import NBAPlayerAnalyzer

# Initialize analyzer
analyzer = NBAPlayerAnalyzer(csv_path='players.csv', min_minutes=200)

# Run full analysis
results = analyzer.run_full_analysis(n_clusters=6, cluster_method='kmeans')
```

### Step-by-Step Usage

You can also run the analysis step by step for more control:

```python
analyzer = NBAPlayerAnalyzer(csv_path='players.csv', min_minutes=200)

# Load and clean data
analyzer.load_data()
analyzer.clean_data()

# Normalize (z-score or minmax)
analyzer.normalize_data(method='zscore')

# Perform PCA
analyzer.perform_pca(n_components=2)  # or None to auto-select

# Interpret components
analyzer.interpret_components()

# Visualize variance
analyzer.visualize_variance()

# Find optimal number of clusters
optimal_k, scores, inertias = analyzer.find_optimal_clusters(max_k=10)

# Cluster players
analyzer.cluster_players(n_clusters=optimal_k, method='kmeans')

# Analyze clusters
analyzer.analyze_clusters()

# Visualize results
analyzer.visualize_pca(show_clusters=True)

# Find similar players
analyzer.find_similar_players("LeBron James", n_similar=10)

# Save results
analyzer.save_results('results.csv')
```

## Key Features

- **Data Cleaning**: Filters players by minimum minutes, handles missing values
- **Normalization**: Z-score standardization or min-max scaling
- **PCA**: Automatic component selection based on variance explained
- **Component Interpretation**: Shows which statistics contribute most to each component
- **Clustering**: K-Means or K-Medoids clustering with optimal k selection
- **Visualization**: 2D/3D PCA plots with cluster assignments
- **Similar Player Finder**: Find players most similar to a given player

## Output Files

- `pca_visualization.png`: 2D/3D scatter plot of players in PCA space
- `variance_analysis.png`: Scree plot and cumulative variance plot
- `nba_analysis_results.csv`: Results with cluster assignments and PCA coordinates

## Parameters

- `min_minutes`: Minimum minutes played to include a player (default: 200)
- `n_clusters`: Number of clusters (or None to find optimal)
- `cluster_method`: 'kmeans' or 'kmedoids'
- `normalize_method`: 'zscore' or 'minmax'

