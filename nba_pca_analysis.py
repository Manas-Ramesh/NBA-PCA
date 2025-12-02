"""
NBA Player Clustering using PCA and K-Means
Also will use KNN and KMedoids for clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, KMedoids
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')




# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class NBAPlayerAnalyzer:
    """
    Main class for NBA player analysis using PCA and clustering
    """
    
    def __init__(self, csv_path='players.csv', min_minutes=200):
        """
        Initialize the analyzer
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file with player data
        min_minutes : int
            Minimum minutes played to include a player (filters outliers)
        """
        self.csv_path = csv_path
        self.min_minutes = min_minutes
        self.df = None
        self.df_clean = None
        self.features = None
        self.X_scaled = None
        self.pca = None
        self.X_pca = None
        self.scaler = None
        self.clusters = None
        
    def load_data(self):
        """Load the CSV data"""
        print("Loading data...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} player-season records")
        return self.df
    
    def clean_data(self):
        """
        Clean the data:
        - Filter by minimum minutes
        - Handle missing values
        - Select relevant features
        """
        print("\nCleaning data...")
        
        # Filter by minimum minutes
        self.df_clean = self.df[self.df['mp'] >= self.min_minutes].copy()
        print(f"After filtering (min {self.min_minutes} minutes): {len(self.df_clean)} records")
        
        # Select numeric features for analysis
        # Exclude: season, player_id, player, team, pos (categorical)
        # Exclude: g (games) - we have minutes which is better
        # Exclude redundant features (e.g., if we have e_fg_percent, we might exclude fg_percent)
        
        feature_columns = [
            # Per-36 stats (core performance metrics)
            'pts_per_36_min',
            'trb_per_36_min',
            'ast_per_36_min',
            'stl_per_36_min',
            'blk_per_36_min',
            'tov_per_36_min',
            'pf_per_36_min',
            
            # Shooting efficiency
            'e_fg_percent',  # Effective field goal percentage (includes 3P value)
            'x3p_percent',   # 3-point percentage
            
            # Shot location and style
            'avg_dist_fga',  # Average distance of field goal attempts
            'percent_fga_from_x0_3_range',  # % of shots from 0-3 feet (rim)
            'percent_fga_from_x3p_range',  # % of shots from 3-point range
            'percent_dunks_of_fga',  # % of shots that are dunks
            
            # Advanced metrics
            'net_plus_minus_per_100_poss',  # Net rating
            'points_generated_by_assists',  # Playmaking impact
        ]
        
        # Check which columns exist
        available_features = [col for col in feature_columns if col in self.df_clean.columns]
        self.features = available_features
        
        print(f"Selected {len(self.features)} features for analysis")
        
        # Extract feature matrix
        X = self.df_clean[self.features].copy()
        
        # Handle missing values
        # For percentages, fill with 0 if missing (means they don't take those shots)
        # For 3P%, fill with 0 if missing
        X = X.fillna(0)
        
        # Check for any remaining missing values
        if X.isnull().sum().sum() > 0:
            print("Warning: Still have missing values, filling with 0")
            X = X.fillna(0)
        
        self.df_clean['feature_matrix'] = [X.iloc[i].values for i in range(len(X))]
        
        print(f"Data cleaning complete. {len(self.df_clean)} players ready for analysis.")
        return self.df_clean
    
    def normalize_data(self, method='zscore'):
        """
        Normalize the feature data
        
        Parameters:
        -----------
        method : str
            'zscore' for standardization (mean=0, std=1)
            'minmax' for min-max scaling (0-1 range)
        """
        print(f"\nNormalizing data using {method}...")
        
        X = self.df_clean[self.features].values
        
        if method == 'zscore':
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(X)
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            self.X_scaled = self.scaler.fit_transform(X)
        else:
            raise ValueError("Method must be 'zscore' or 'minmax'")
        
        print(f"Normalization complete. Shape: {self.X_scaled.shape}")
        return self.X_scaled
    
    def perform_pca(self, n_components=None):
        """
        Perform Principal Component Analysis
        
        Parameters:
        -----------
        n_components : int or None
            Number of components to keep. If None, keeps all for analysis
        """
        print("\nPerforming PCA...")
        
        if self.X_scaled is None:
            raise ValueError("Must normalize data first!")
        
        # If n_components not specified, fit with all components first
        if n_components is None:
            self.pca = PCA()
            self.pca.fit(self.X_scaled)
            
            # Calculate cumulative variance
            cumsum_variance = np.cumsum(self.pca.explained_variance_ratio_)
            
            # Find number of components for 80% variance
            n_80 = np.argmax(cumsum_variance >= 0.80) + 1
            n_90 = np.argmax(cumsum_variance >= 0.90) + 1
            
            print(f"\nVariance explained by components:")
            print(f"  First 2 components: {cumsum_variance[1]:.1%}")
            print(f"  First 3 components: {cumsum_variance[2]:.1%}")
            print(f"  Components for 80% variance: {n_80}")
            print(f"  Components for 90% variance: {n_90}")
            
            # Use 2 or 3 components (whichever explains more variance, but prioritize 2D visualization)
            if cumsum_variance[1] >= 0.50:  # If 2 components explain at least 50%
                n_components = 2
                print(f"\nUsing 2 components for visualization ({cumsum_variance[1]:.1%} variance)")
            else:
                n_components = 3
                print(f"\nUsing 3 components ({cumsum_variance[2]:.1%} variance)")
        
        # Fit PCA with selected number of components
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        print(f"\nPCA complete:")
        print(f"  Components: {n_components}")
        print(f"  Variance explained: {self.pca.explained_variance_ratio_.sum():.1%}")
        print(f"  Individual component variance:")
        for i, var in enumerate(self.pca.explained_variance_ratio_, 1):
            print(f"    PC{i}: {var:.1%}")
        
        return self.X_pca, self.pca
    
    def interpret_components(self, top_n=5):
        """
        Interpret what each principal component represents
        
        Parameters:
        -----------
        top_n : int
            Number of top features to show for each component
        """
        print("\n" + "="*60)
        print("PRINCIPAL COMPONENT INTERPRETATION")
        print("="*60)
        
        components = self.pca.components_
        feature_names = self.features
        
        for i, component in enumerate(components, 1):
            print(f"\nPrincipal Component {i} (explains {self.pca.explained_variance_ratio_[i-1]:.1%} variance):")
            
            # Get top contributing features (positive and negative)
            feature_contributions = list(zip(feature_names, component))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print("  Top contributing features:")
            for feat, weight in feature_contributions[:top_n]:
                direction = "+" if weight > 0 else "-"
                print(f"    {direction} {feat:30s} ({weight:+.3f})")
    
    def find_optimal_clusters(self, max_k=10, method='kmeans'):
        """
        Find optimal number of clusters using silhouette score
        
        Parameters:
        -----------
        max_k : int
            Maximum number of clusters to test
        method : str
            'kmeans' or 'kmedoids'
        """
        print(f"\nFinding optimal number of clusters ({method})...")
        
        if self.X_pca is None:
            raise ValueError("Must perform PCA first!")
        
        silhouette_scores = []
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            elif method == 'kmedoids':
                from sklearn_extra.cluster import KMedoids
                clusterer = KMedoids(n_clusters=k, random_state=42)
            else:
                raise ValueError("Method must be 'kmeans' or 'kmedoids'")
            
            cluster_labels = clusterer.fit_predict(self.X_pca)
            silhouette_avg = silhouette_score(self.X_pca, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            
            if method == 'kmeans':
                inertias.append(clusterer.inertia_)
        
        # Find optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        print(f"\nSilhouette scores:")
        for k, score in zip(k_range, silhouette_scores):
            marker = " <-- OPTIMAL" if k == optimal_k else ""
            print(f"  k={k}: {score:.3f}{marker}")
        
        return optimal_k, silhouette_scores, inertias if method == 'kmeans' else None
    
    def cluster_players(self, n_clusters, method='kmeans'):
        """
        Cluster players using specified method
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        method : str
            'kmeans' or 'kmedoids'
        """
        print(f"\nClustering players into {n_clusters} clusters using {method}...")
        
        if self.X_pca is None:
            raise ValueError("Must perform PCA first!")
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'kmedoids':
            from sklearn_extra.cluster import KMedoids
            clusterer = KMedoids(n_clusters=n_clusters, random_state=42)
        else:
            raise ValueError("Method must be 'kmeans' or 'kmedoids'")
        
        self.clusters = clusterer.fit_predict(self.X_pca)
        self.df_clean['cluster'] = self.clusters
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.X_pca, self.clusters)
        print(f"Clustering complete. Silhouette score: {silhouette_avg:.3f}")
        
        # Show cluster sizes
        cluster_counts = pd.Series(self.clusters).value_counts().sort_index()
        print(f"\nCluster sizes:")
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id}: {count} players")
        
        return self.clusters
    
    def visualize_pca(self, show_clusters=True):
        """
        Visualize PCA results
        
        Parameters:
        -----------
        show_clusters : bool
            Whether to color by cluster assignments
        """
        print("\nCreating PCA visualization...")
        
        if self.X_pca is None:
            raise ValueError("Must perform PCA first!")
        
        n_components = self.X_pca.shape[1]
        
        if n_components == 2:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            if show_clusters and self.clusters is not None:
                scatter = ax.scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
                                   c=self.clusters, cmap='tab10', 
                                   alpha=0.6, s=50)
                plt.colorbar(scatter, ax=ax, label='Cluster')
            else:
                ax.scatter(self.X_pca[:, 0], self.X_pca[:, 1], alpha=0.6, s=50)
            
            ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            ax.set_title('NBA Players in 2D PCA Space', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
        elif n_components >= 3:
            fig = plt.figure(figsize=(15, 5))
            
            # 2D projections
            projections = [(0, 1), (0, 2), (1, 2)]
            titles = ['PC1 vs PC2', 'PC1 vs PC3', 'PC2 vs PC3']
            
            for idx, (i, j) in enumerate(projections):
                ax = fig.add_subplot(1, 3, idx + 1)
                
                if show_clusters and self.clusters is not None:
                    scatter = ax.scatter(self.X_pca[:, i], self.X_pca[:, j], 
                                       c=self.clusters, cmap='tab10', 
                                       alpha=0.6, s=50)
                    if idx == 0:
                        plt.colorbar(scatter, ax=ax, label='Cluster')
                else:
                    ax.scatter(self.X_pca[:, i], self.X_pca[:, j], alpha=0.6, s=50)
                
                ax.set_xlabel(f'PC{i+1} ({self.pca.explained_variance_ratio_[i]:.1%})', fontsize=10)
                ax.set_ylabel(f'PC{j+1} ({self.pca.explained_variance_ratio_[j]:.1%})', fontsize=10)
                ax.set_title(titles[idx], fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
        print("Saved visualization to 'pca_visualization.png'")
        plt.show()
    
    def visualize_variance(self):
        """Create scree plot and cumulative variance plot"""
        print("\nCreating variance plots...")
        
        # Fit PCA with all components to see full variance breakdown
        pca_full = PCA()
        pca_full.fit(self.X_scaled)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scree plot
        components = range(1, len(pca_full.explained_variance_ratio_) + 1)
        ax1.bar(components, pca_full.explained_variance_ratio_)
        ax1.plot(components, pca_full.explained_variance_ratio_, 'ro-', linewidth=2)
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Variance Explained', fontsize=12)
        ax1.set_title('Scree Plot', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative variance plot
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        ax2.plot(components, cumsum_variance, 'bo-', linewidth=2, markersize=8)
        ax2.axhline(y=0.8, color='r', linestyle='--', label='80% Variance')
        ax2.axhline(y=0.9, color='g', linestyle='--', label='90% Variance')
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Variance Explained', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('variance_analysis.png', dpi=300, bbox_inches='tight')
        print("Saved variance plots to 'variance_analysis.png'")
        plt.show()
    
    def analyze_clusters(self, top_n=10):
        """
        Analyze cluster characteristics
        
        Parameters:
        -----------
        top_n : int
            Number of players to show per cluster
        """
        print("\n" + "="*60)
        print("CLUSTER ANALYSIS")
        print("="*60)
        
        if self.clusters is None:
            print("No clusters assigned yet!")
            return
        
        for cluster_id in sorted(self.df_clean['cluster'].unique()):
            cluster_players = self.df_clean[self.df_clean['cluster'] == cluster_id]
            
            print(f"\n{'='*60}")
            print(f"CLUSTER {cluster_id} ({len(cluster_players)} players)")
            print(f"{'='*60}")
            
            # Average stats for this cluster
            print("\nAverage statistics:")
            avg_stats = cluster_players[self.features].mean()
            for feat, val in avg_stats.items():
                print(f"  {feat:30s}: {val:8.2f}")
            
            # Top players in this cluster
            print(f"\nSample players (showing {min(top_n, len(cluster_players))}):")
            for idx, row in cluster_players.head(top_n).iterrows():
                print(f"  - {row['player']:25s} ({row['team']:3s}, {row['pos']})")
    
    def find_similar_players(self, player_name, n_similar=10):
        """
        Find players similar to a given player
        
        Parameters:
        -----------
        player_name : str
            Name of the player to find similarities for
        n_similar : int
            Number of similar players to return
        """
        print(f"\nFinding players similar to {player_name}...")
        
        if self.X_pca is None:
            raise ValueError("Must perform PCA first!")
        
        # Find the player
        player_mask = self.df_clean['player'].str.contains(player_name, case=False, na=False)
        matching_players = self.df_clean[player_mask]
        
        if len(matching_players) == 0:
            print(f"Player '{player_name}' not found!")
            return None
        
        if len(matching_players) > 1:
            print(f"Multiple players found. Using first match: {matching_players.iloc[0]['player']}")
        
        player_idx = matching_players.index[0]
        player_data = matching_players.iloc[0]
        player_pca = self.X_pca[player_idx]
        
        # Calculate distances to all other players
        distances = np.sqrt(((self.X_pca - player_pca) ** 2).sum(axis=1))
        
        # Get similar players (excluding the player themselves)
        similar_indices = np.argsort(distances)[1:n_similar+1]  # Skip index 0 (the player themselves)
        
        print(f"\nMost similar players to {player_data['player']} ({player_data['team']}, {player_data['pos']}):")
        print(f"Cluster: {player_data['cluster']}")
        print("\nSimilar players:")
        
        for i, idx in enumerate(similar_indices, 1):
            similar_player = self.df_clean.iloc[idx]
            distance = distances[idx]
            print(f"  {i:2d}. {similar_player['player']:25s} ({similar_player['team']:3s}, {similar_player['pos']:2s}) "
                  f"[Cluster {similar_player['cluster']}, Distance: {distance:.3f}]")
        
        return similar_indices
    
    def save_results(self, filename='nba_analysis_results.csv'):
        """Save results to CSV"""
        print(f"\nSaving results to {filename}...")
        
        # Add PCA coordinates to dataframe
        if self.X_pca is not None:
            for i in range(self.X_pca.shape[1]):
                self.df_clean[f'PC{i+1}'] = self.X_pca[:, i]
        
        # Save
        output_cols = ['season', 'player', 'team', 'pos', 'cluster'] + \
                     [f'PC{i+1}' for i in range(self.X_pca.shape[1]) if self.X_pca is not None] + \
                     self.features
        
        self.df_clean[output_cols].to_csv(filename, index=False)
        print(f"Results saved!")
    
    def run_full_analysis(self, n_clusters=None, cluster_method='kmeans', normalize_method='zscore'):
        """
        Run the complete analysis pipeline
        
        Parameters:
        -----------
        n_clusters : int or None
            Number of clusters. If None, will find optimal
        cluster_method : str
            'kmeans' or 'kmedoids'
        normalize_method : str
            'zscore' or 'minmax'
        """
        print("="*60)
        print("NBA PLAYER PCA ANALYSIS - FULL PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Clean data
        self.clean_data()
        
        # Step 3: Normalize
        self.normalize_data(method=normalize_method)
        
        # Step 4: PCA
        self.perform_pca()
        
        # Step 5: Interpret components
        self.interpret_components()
        
        # Step 6: Visualize variance
        self.visualize_variance()
        
        # Step 7: Find optimal clusters (if not specified)
        if n_clusters is None:
            optimal_k, scores, inertias = self.find_optimal_clusters(method=cluster_method)
            n_clusters = optimal_k
        
        # Step 8: Cluster players
        self.cluster_players(n_clusters, method=cluster_method)
        
        # Step 9: Analyze clusters
        self.analyze_clusters()
        
        # Step 10: Visualize
        self.visualize_pca(show_clusters=True)
        
        # Step 11: Save results
        self.save_results()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
        return self.df_clean


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = NBAPlayerAnalyzer(csv_path='players.csv', min_minutes=200)
    
    # Run full analysis
    results = analyzer.run_full_analysis(n_clusters=6, cluster_method='kmeans')
    
    # Example: Find similar players
    # analyzer.find_similar_players("LeBron James", n_similar=10)
    # analyzer.find_similar_players("Stephen Curry", n_similar=10)

