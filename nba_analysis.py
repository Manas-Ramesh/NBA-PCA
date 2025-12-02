"""
NBA Player Clustering using PCA and K-Means
Also will use KNN and KMedoids for clustering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class NBADataLoader:
    def __init__(self, csv_path='players.csv'):
        self.csv_path = csv_path
        self.df = None
        self.df_clean = None
        self.features = None
        self.X = None
        self.X_scaled = None
        self.scaler = None
        self.pca = None
        self.X_pca = None
        self.clusters = None
        self.kmeans = None

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.df)} player-season records")
        return self.df
    def clean_data(self, min_minutes=200):
        if self.df is None:
            print("load first")
            return None


        required_minutes = self.df['mp'] >= min_minutes
        filtered_df = self.df[required_minutes]
        self.df_clean = filtered_df.copy()
        
        print(f"\nMissing values before cleaning:")
        missing_before = self.df_clean.isnull().sum()
        print(missing_before[missing_before > 0])
        
        #for missing values, fill with 0
        self.df_clean = self.df_clean.fillna(0)
        missing_after = self.df_clean.isnull().sum().sum()
        print(f"\nMissing values after cleaning: {missing_after}")
        
        return self.df_clean
    
    def stats_data(self):
        if self.df_clean is None:
            print("Error: Must clean data first!")
            return None

        features = ['pts_per_36_min', 'trb_per_36_min','ast_per_36_min','stl_per_36_min', 'blk_per_36_min','tov_per_36_min',
                    'pf_per_36_min', 'e_fg_percent','x3p_percent', 'avg_dist_fga', 'percent_fga_from_x0_3_range', 'percent_fga_from_x3p_range',
                    'percent_dunks_of_fga', 'net_plus_minus_per_100_poss', 'points_generated_by_assists']
        
        available_features = []
        missing_features = []
        
        for stat in features:
            if stat in self.df_clean.columns:
                available_features.append(stat)
            else:
                missing_features.append(stat)
        
        if missing_features:
            print(f"not found:{missing_features}")
        
        self.features = available_features
        print(f"Selected stats: {len(self.features)}")
        for i, stat in enumerate(self.features, 1):
            print(f"  {i:2d}. {stat}")
        
        self.X = self.df_clean[self.features].values
        print(f"\nFeature matrix shape: {self.X.shape} (players by features)")
        
        return self.features, self.X
    
    def normalize_data(self):
        if self.X is None:
            print("Select features first")
            return None
        for i, stat in enumerate(self.features):
            min_val = self.X[:, i].min()
            max_val = self.X[:, i].max()
            mean_val = self.X[:, i].mean()
            std_val = self.X[:, i].std()
            print(f"  {stat:30s}: [{min_val:7.2f}, {max_val:7.2f}] mean={mean_val:7.2f}, std={std_val:7.2f}")
        
        # z-score normalization
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        

        for i, feat in enumerate(self.features):
            mean_val = self.X_scaled[:, i].mean()
            std_val = self.X_scaled[:, i].std()
            print(f"  {feat:30s}: mean={mean_val:7.3f}, std={std_val:7.3f}")
        
        print(f"\nNormalized shape: {self.X_scaled.shape}")
        
        return self.X_scaled
    
    def pcaAnalysis(self, n_components=None):
        if self.X_scaled is None:
            print("Error: Must normalize data first!")
            return None
        
        pca_full = PCA()
        pca_full.fit(self.X_scaled)
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        print("Variance explained by components:")
        for i in range(min(5, len(cumsum_variance))):
            var_explained = pca_full.explained_variance_ratio_[i]
            cum_var = cumsum_variance[i]
            print(f"  PC{i+1}: {var_explained:.1%} variance (cumulative: {cum_var:.1%})")
        
        if n_components is None:
            if cumsum_variance[1] >= 0.50:
                n_components = 2
                print(f"\nUsing 2 components ({cumsum_variance[1]:.1%} variance explained)")
            else:
                n_components = 3
                print(f"\nUsing 3 components ({cumsum_variance[2]:.1%} variance explained)")
        
        #components
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        print(f"  Reduced from {self.X_scaled.shape[1]} features to {n_components} components")
        print(f"  Total variance explained: {self.pca.explained_variance_ratio_.sum():.1%}")
        print(f"  PCA data shape: {self.X_pca.shape} (players by components)")
        
        return self.X_pca, self.pca
    

    def find_optimal_k(self, max_k=10):
        if self.X_pca is None:
            print("do PCA first")
            return None      
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X_pca)
            silhouette_avg = silhouette_score(self.X_pca, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        print(f"\nSilhouette scores")
        for k, score in zip(k_range, silhouette_scores):
            marker = " <-- OPTIMAL" if k == optimal_k else ""
            print(f"  k={k}: {score:.3f}{marker}")
        
        return optimal_k, silhouette_scores
    
    def cluster_players(self, n_clusters):

        if self.X_pca is None:
            print("do pca first")
            return None
        
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = self.kmeans.fit_predict(self.X_pca)
        
        self.df_clean['cluster'] = self.clusters
        
        silhouette_avg = silhouette_score(self.X_pca, self.clusters)
        print(f"Silhouette score: {silhouette_avg:.3f}")
        

        cluster_counts = pd.Series(self.clusters).value_counts().sort_index()
        print(f"\nCluster sizes:")
        for cluster_id, count in cluster_counts.items():
            print(f"  Cluster {cluster_id}: {count} players")
        
        return self.clusters


if __name__ == "__main__":
    # Step 1: Load data
    loader = NBADataLoader(csv_path='players.csv')
    df = loader.load_data()
    print("First few rows of the data:")
    print(df.head())
    print("\nmissing if any:")
    print(df.isnull().sum())
    
    # Step 2: Clean data
    df_clean = loader.clean_data(min_minutes=200)
    print(f"\nCleaned data shape: {df_clean.shape}")
    print(df_clean[['player', 'team', 'pos', 'mp']].head(10))
    
    # Step 3: Select features
    features, X = loader.stats_data()
    print(f"\nSample of feature values (first 3 players, first 5 features):")
    print(X[:3, :5])
    
    # Step 4: Normalize data
    X_scaled = loader.normalize_data()
    print(f"\nSample of normalized values (first 3 players, first 5 features):")
    print(X_scaled[:3, :5])
    
    # Step 5: Perform PCA
    X_pca, pca = loader.pcaAnalysis(n_components=3)
    print(f"\nSample of PCA values (first 3 players, all components):")
    print(X_pca[:3, :])
    
    # Step 7: Find optimal k and cluster players
    optimal_k, scores = loader.find_optimal_k(max_k=10)
    clusters = loader.cluster_players(n_clusters=optimal_k)
    
    # Show some players from each cluster
    print(f"\nSample players from each cluster:")
    for cluster_id in sorted(loader.df_clean['cluster'].unique()):
        cluster_players = loader.df_clean[loader.df_clean['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_players)} players):")
        print(cluster_players[['player', 'team', 'pos']].head(5).to_string(index=False))
