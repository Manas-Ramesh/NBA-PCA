"""
Linear Algebra Project
Manas Ramesh and Isaac Lehrer

NBA Player Clustering using PCA and K-Means
For future reference, maybe use KNN later?
And add more stats with more features?
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

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
        
        self.df_clean = self.df_clean.fillna(0)
        missing_after = self.df_clean.isnull().sum().sum()
        
        return self.df_clean
    
    def stats_data(self):
        if self.df_clean is None:
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
        
        self.features = available_features
        self.X = self.df_clean[self.features].values
        
        print(f"\n{'='*70}")
        print("FEATURE MATRIX STATISTICS")
        print(f"{'='*70}")
        print(f"Matrix shape: {self.X.shape} (players × features)")
        print(f"Total players: {self.X.shape[0]}")
        print(f"Total features: {self.X.shape[1]}")
        print(f"\nFeature matrix summary:")
        print(f"  Min values: {self.X.min(axis=0)}")
        print(f"  Max values: {self.X.max(axis=0)}")
        print(f"  Mean values: {self.X.mean(axis=0)}")
        print(f"  Std values: {self.X.std(axis=0)}")
        print(f"\nSample of first 3 players, first 5 features:")
        print(self.X[:3, :5])
        
        return self.features, self.X
    
    def normalize_data(self):
        if self.X is None:
            print("Select features first")
            return None
        
        print(f"\n{'='*70}")
        print("NORMALIZATION: BEFORE → AFTER")
        print(f"{'='*70}")
        print(f"{'Feature':30s} | {'Before':>25s} | {'After':>25s}")
        print(f"{'-'*30} | {'-'*25} | {'-'*25}")
        
        for i, stat in enumerate(self.features):
            min_val_before = self.X[:, i].min()
            max_val_before = self.X[:, i].max()
            mean_val_before = self.X[:, i].mean()
            std_val_before = self.X[:, i].std()
            before_str = f"mean={mean_val_before:7.2f}, std={std_val_before:7.2f}"

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        for i, feat in enumerate(self.features):
            mean_val_after = self.X_scaled[:, i].mean()
            std_val_after = self.X_scaled[:, i].std()
            after_str = f"mean={mean_val_after:7.3f}, std={std_val_after:7.3f}"
            min_val_before = self.X[:, i].min()
            max_val_before = self.X[:, i].max()
            mean_val_before = self.X[:, i].mean()
            std_val_before = self.X[:, i].std()
            before_str = f"mean={mean_val_before:7.2f}, std={std_val_before:7.2f}"
            print(f"  {feat:30s} | {before_str:>25s} | {after_str:>25s}")
        
        print(f"\nNormalized matrix shape: {self.X_scaled.shape}")
        print(f"Sample normalized values (first 3 players, first 5 features):")
        print(self.X_scaled[:3, :5])
        
        return self.X_scaled
    
    def pcaAnalysis(self, n_components=None):
        if self.X_scaled is None:
            return None
        
        print(f"\n{'='*70}")
        print("PCA TRANSFORMATION")
        print(f"{'='*70}")
        print(f"Input matrix shape: {self.X_scaled.shape} (players × features)")
        
        pca_full = PCA()
        pca_full.fit(self.X_scaled)
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        print(f"\nVariance explained by components:")
        for i in range(min(5, len(cumsum_variance))):
            var_explained = pca_full.explained_variance_ratio_[i]
            cum_var = cumsum_variance[i]
            print(f"  PC{i+1}: {var_explained:.1%} variance (cumulative: {cum_var:.1%})")
        
        if n_components is None:
            if cumsum_variance[1] >= 0.50:
                n_components = 2
            else:
                n_components = 3
        
        print(f"\nUsing {n_components} components")
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        print(f"Output matrix shape: {self.X_pca.shape} (players × components)")
        print(f"Total variance explained: {self.pca.explained_variance_ratio_.sum():.1%}")
        print(f"\nPCA transformation sample (first 3 players, all components):")
        print(self.X_pca[:3, :])
        print(f"\nPCA component ranges:")
        for i in range(n_components):
            print(f"  PC{i+1}: [{self.X_pca[:, i].min():.2f}, {self.X_pca[:, i].max():.2f}]")
        
        return self.X_pca, self.pca
    

    def find_optimal_k(self, max_k=10):
        if self.X_pca is None:
            print("do PCA first")
            return None      
        
        print(f"\n{'='*70}")
        print(f"FINDING OPTIMAL K (testing k=2 to k={max_k})")
        print(f"{'='*70}")
        
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X_pca)
            silhouette_avg = silhouette_score(self.X_pca, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        print(f"\nSilhouette scores:")
        for k, score in zip(k_range, silhouette_scores):
            if k == optimal_k:
                marker = " <-- OPTIMAL"
            else:
                marker = ""
            print(f"  k={k:2d}: {score:.4f}{marker}")
        
        print(f"\nOptimal k: {optimal_k} (silhouette score: {silhouette_scores[optimal_k-2]:.4f})")
        
        return optimal_k, silhouette_scores
    
    def cluster_players(self, n_clusters):
        if self.X_pca is None:
            return None
        
        print(f"\n{'='*70}")
        print(f"CLUSTERING: K-Means with k={n_clusters}")
        print(f"{'='*70}")
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = self.kmeans.fit_predict(self.X_pca)
        
        self.df_clean['cluster'] = self.clusters
        
        silhouette_avg = silhouette_score(self.X_pca, self.clusters)
        cluster_counts = pd.Series(self.clusters).value_counts().sort_index()
        
        print(f"Silhouette score: {silhouette_avg:.4f}")
        print(f"\nCluster sizes:")
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(self.clusters)) * 100
            print(f"  Cluster {cluster_id:2d}: {count:4d} players ({percentage:5.1f}%)")
        
        print(f"\nCluster centroids in PCA space:")
        for i, centroid in enumerate(self.kmeans.cluster_centers_):
            print(f"  Cluster {i:2d}: PC1={centroid[0]:7.2f}, PC2={centroid[1]:7.2f}, PC3={centroid[2]:7.2f}")
        
        return self.clusters
    
    def analyze_clusters(self):
        if self.clusters is None:
            return None
        
        print(f"\n{'='*70}")
        print("CLUSTER ANALYSIS: Average Statistics")
        print(f"{'='*70}")
        
        for cluster_id in sorted(self.df_clean['cluster'].unique()):
            cluster_players = self.df_clean[self.df_clean['cluster'] == cluster_id]
            avg_stats = cluster_players[self.features].mean()
            top_stats = avg_stats.nlargest(5)
            
            print(f"\nCluster {cluster_id} ({len(cluster_players)} players):")
            print("-" * 60)
            print("Top 5 average stats:")
            for stat, val in top_stats.items():
                print(f"  {stat:30s}: {val:8.2f}")
    
    def visualize_clusters(self, save_plot=True):
        if self.X_pca is None or self.clusters is None:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        projections = [
            (0, 1, 'PC1', 'PC2'),
            (0, 2, 'PC1', 'PC3'),
            (1, 2, 'PC2', 'PC3')
        ]
        
        var_explained = self.pca.explained_variance_ratio_
        
        for idx, (i, j, label_i, label_j) in enumerate(projections):
            ax = axes[idx]
            
            scatter = ax.scatter(
                self.X_pca[:, i], 
                self.X_pca[:, j], 
                c=self.clusters, 
                cmap='tab20', 
                alpha=0.6, 
                s=10
            )
            
            ax.set_xlabel(f'{label_i} ({var_explained[i]:.1%} variance)', fontsize=11)
            ax.set_ylabel(f'{label_j} ({var_explained[j]:.1%} variance)', fontsize=11)
            ax.set_title(f'{label_i} vs {label_j}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.suptitle('NBA Players Clustered in PCA Space', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('nba_clusters_visualization.png', dpi=300, bbox_inches='tight')
            print("Saved visualization to 'nba_clusters_visualization.png'")
        
        plt.show()
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
        
        scatter2 = ax2.scatter(
            self.X_pca[:, 0], 
            self.X_pca[:, 1], 
            c=self.clusters, 
            cmap='tab20', 
            alpha=0.6, 
            s=15
        )
        
        ax2.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)', fontsize=12)
        ax2.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)', fontsize=12)
        ax2.set_title('NBA Players Clustered in 2D PCA Space (PC1 vs PC2)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
        
        if save_plot:
            plt.savefig('nba_clusters_pc1_pc2.png', dpi=300, bbox_inches='tight')
            print("Saved main visualization to 'nba_clusters_pc1_pc2.png'")
        
        plt.show()
        
        print("Visualization complete!")
    
    def visualize_clusters_3d(self, save_plot=True):
        if self.X_pca is None or self.clusters is None:
            print("do pca and cluster first")
            return None
        
        if self.X_pca.shape[1] < 3:
            print("Need at least 3 PCA components for 3D visualization")
            return None
        
        var_explained = self.pca.explained_variance_ratio_
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            self.X_pca[:, 0],
            self.X_pca[:, 1],
            self.X_pca[:, 2],
            c=self.clusters,
            cmap='tab20',
            alpha=0.6,
            s=20
        )
        
        ax.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)', fontsize=12)
        ax.set_zlabel(f'PC3 ({var_explained[2]:.1%} variance)', fontsize=12)
        ax.set_title('NBA Players Clustered in 3D PCA Space', fontsize=14, fontweight='bold', pad=20)
        
        plt.colorbar(scatter, ax=ax, label='Cluster', shrink=0.8)
        
        if save_plot:
            plt.savefig('nba_clusters_3d.png', dpi=300, bbox_inches='tight')
            print("Saved 3D visualization to 'nba_clusters_3d.png'")
        
        plt.show()
        
        print("3D visualization complete!")
    
    def visualize_cluster_centroids_radar(self, save_plot=True, n_features=8):

        if self.clusters is None:
            print("cluster first")
            return None

        cluster_centroids = {}
        for cluster_id in sorted(self.df_clean['cluster'].unique()):
            cluster_data = self.df_clean[self.df_clean['cluster'] == cluster_id]
            centroids = cluster_data[self.features].mean()
            cluster_centroids[cluster_id] = centroids
        
        feature_variance = {}
        for feat in self.features:
            values = [cluster_centroids[c][feat] for c in cluster_centroids.keys()]
            feature_variance[feat] = np.var(values)
        
        top_features = sorted(feature_variance.items(), key=lambda x: x[1], reverse=True)[:n_features]
        top_feature_names = [feat[0] for feat in top_features]
        
        print(f"Using top {n_features} most variable features: {top_feature_names}")
        
        normalized_centroids = {}
        for cluster_id in cluster_centroids.keys():
            normalized = {}
            for feat in top_feature_names:
                all_values = []
                for c in cluster_centroids.keys():
                    all_values.append(cluster_centroids[c][feat])
                min_val = min(all_values)
                max_val = max(all_values)
                if max_val != min_val:
                    normalized[feat] = (cluster_centroids[cluster_id][feat] - min_val) / (max_val - min_val)
                else:
                    normalized[feat] = 0.5
            normalized_centroids[cluster_id] = normalized
        
        n_clusters = len(cluster_centroids)
        n_cols = 5  
        n_rows = (n_clusters + n_cols - 1) // n_cols  
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows), subplot_kw=dict(projection='polar'))
        axes = axes.flatten() if n_clusters > 1 else [axes]
        
        N = len(top_feature_names)
        
        angles = []
        for n in range(N):
            angles.append(n / float(N) * 2 * np.pi)
        angles.append(angles[0])
        
        for idx, cluster_id in enumerate(sorted(cluster_centroids.keys())):
            ax = axes[idx]
            
            values = []
            for feat in top_feature_names:
                values.append(normalized_centroids[cluster_id][feat])
            values.append(values[0])
            
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}')
            ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            short_names = []
            for name in top_feature_names:
                short_name = name.replace('_per_36_min', '/36').replace('percent_', '%').replace('_', ' ')[:15]
                short_names.append(short_name)
            ax.set_xticklabels(short_names, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
            ax.grid(True)
            
            cluster_size = len(self.df_clean[self.df_clean['cluster'] == cluster_id])
            ax.set_title(f'Cluster {cluster_id}\n({cluster_size} players)', 
                        fontsize=10, fontweight='bold', pad=20)
        
        # Hide unused subplots
        for idx in range(n_clusters, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Cluster Centroids - Radar Charts\n(Normalized feature values)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('nba_cluster_centroids_radar.png', dpi=300, bbox_inches='tight')
            print("Saved radar charts to 'nba_cluster_centroids_radar.png'")
        
        plt.show()
        
        fig2, ax2 = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        
        for idx, cluster_id in enumerate(sorted(cluster_centroids.keys())):
            values = []
            for feat in top_feature_names:
                values.append(normalized_centroids[cluster_id][feat])
            values.append(values[0])
            
            ax2.plot(angles, values, 'o-', linewidth=1.5, label=f'Cluster {cluster_id}', 
                    color=colors[idx], alpha=0.7)
            ax2.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax2.set_xticks(angles[:-1])
        short_names = []
        for name in top_feature_names:
            short_name = name.replace('_per_36_min', '/36').replace('percent_', '%').replace('_', ' ')[:20]
            short_names.append(short_name)
        ax2.set_xticklabels(short_names, fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax2.grid(True)
        ax2.set_title('All Cluster Centroids Comparison\n(Normalized feature values)', 
                     fontsize=14, fontweight='bold', pad=30)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        
        if save_plot:
            plt.savefig('nba_cluster_centroids_comparison.png', dpi=300, bbox_inches='tight')
            print("Saved comparison chart to 'nba_cluster_centroids_comparison.png'")
        
        plt.show()
        
        print("Radar chart visualization complete!")
    
    def find_similar_players(self, player_name, n_similar=10, method='cluster'):

        if self.X_pca is None or self.clusters is None:
            print("cluster first")
            return None
        
        print(f"\n--- Finding Similar Players to '{player_name}' ---")
        
        player_mask = self.df_clean['player'].str.contains(player_name, case=False, na=False)
        matching_players = self.df_clean[player_mask]
        
        if len(matching_players) == 0:
            print(f"Player '{player_name}' not found!")
            print("Try searching with a partial name (e.g., 'LeBron' instead of full name)")
            return None
        
        if len(matching_players) > 1:
            print(f"Multiple players found. Using first match:")
            for idx, row in matching_players.iterrows():
                print(f"  - {row['player']} ({row['team']}, {row['pos']})")
            print(f"\nUsing: {matching_players.iloc[0]['player']}")
        
        player_idx = matching_players.index[0]
        player_data = matching_players.iloc[0]
        player_pca = self.X_pca[player_idx]
        player_cluster = player_data['cluster']
        
        
        print(f"\nPlayer: {player_data['player']}")
        print(f"Team: {player_data['team']}, Position: {player_data['pos']}")
        print(f"Cluster: {player_cluster}")
        print(f"PCA coordinates: [{player_pca[0]:.2f}, {player_pca[1]:.2f}, {player_pca[2]:.2f}]")
        
        results = {}
        
        if method in ['cluster', 'both']:
            cluster_players = self.df_clean[self.df_clean['cluster'] == player_cluster].copy()
            cluster_players = cluster_players[cluster_players.index != player_idx]
            
            if len(cluster_players) > 0:
                cluster_indices = cluster_players.index
                cluster_pca = self.X_pca[cluster_indices]
                distances = np.sqrt(((cluster_pca - player_pca) ** 2).sum(axis=1))
                cluster_players['distance'] = distances
                cluster_players = cluster_players.sort_values('distance')
                results['cluster'] = cluster_players.head(n_similar)
        
        if method in ['distance', 'both']:
            distances = np.sqrt(((self.X_pca - player_pca) ** 2).sum(axis=1))
            similar_indices = np.argsort(distances)[1:n_similar+1]
            similar_players = self.df_clean.iloc[similar_indices].copy()
            similar_players['distance'] = distances[similar_indices]
            results['distance'] = similar_players
        
        if method == 'cluster' or method == 'both':
            print(f"\n{'='*70}")
            print(f"Players in SAME CLUSTER ({player_cluster}) - Similarity Scores:")
            print(f"{'='*70}")
            if len(results['cluster']) > 0:
                print(f"{'Rank':<6} {'Player':<25} {'Team':<5} {'Pos':<4} {'Distance':<10} {'Similarity':<10}")
                print("-" * 70)
                for i, (idx, row) in enumerate(results['cluster'].iterrows(), 1):
                    similarity = 1 / (1 + row['distance'])
                    print(f"{i:<6} {row['player']:<25} {row['team']:<5} {row['pos']:<4} {row['distance']:<10.3f} {similarity:<10.3f}")
            else:
                print("No other players in this cluster.")
        
        if method == 'distance' or method == 'both':
            print(f"\n{'='*70}")
            print(f"Most SIMILAR PLAYERS (by PCA distance) - Similarity Scores:")
            print(f"{'='*70}")
            print(f"{'Rank':<6} {'Player':<25} {'Team':<5} {'Pos':<4} {'Cluster':<10} {'Distance':<10} {'Similarity':<10}")
            print("-" * 80)
            for i, (idx, row) in enumerate(results['distance'].iterrows(), 1):
                cluster_info = f"Cluster {int(row['cluster'])}"
                similarity = 1 / (1 + row['distance'])
                print(f"{i:<6} {row['player']:<25} {row['team']:<5} {row['pos']:<4} {cluster_info:<10} {row['distance']:<10.3f} {similarity:<10.3f}")
        
        return results
    
    def show_cluster_similarities(self):
        if self.clusters is None or self.X_pca is None:
            print("Must cluster players first")
            return None
        
        print(f"\n{'='*70}")
        print("CLUSTER SIMILARITY ANALYSIS")
        print(f"{'='*70}")
        
        n_clusters = len(np.unique(self.clusters))
        cluster_centroids = self.kmeans.cluster_centers_
        
        print(f"\nInter-Cluster Distances (Euclidean distance between centroids):")
        print(f"{'Cluster':<10} ", end="")
        for i in range(n_clusters):
            print(f"{'C'+str(i):<10}", end="")
        print()
        print("-" * (10 + n_clusters * 10))
        
        for i in range(n_clusters):
            print(f"Cluster {i:<6} ", end="")
            for j in range(n_clusters):
                if i == j:
                    dist = 0.0
                else:
                    dist = np.sqrt(((cluster_centroids[i] - cluster_centroids[j]) ** 2).sum())
                print(f"{dist:<10.3f}", end="")
            print()
        
        print(f"\nIntra-Cluster Statistics (within-cluster distances):")
        for cluster_id in sorted(np.unique(self.clusters)):
            cluster_mask = self.clusters == cluster_id
            cluster_pca = self.X_pca[cluster_mask]
            centroid = cluster_centroids[cluster_id]
            
            distances_to_centroid = np.sqrt(((cluster_pca - centroid) ** 2).sum(axis=1))
            avg_distance = distances_to_centroid.mean()
            max_distance = distances_to_centroid.max()
            min_distance = distances_to_centroid.min()
            
            print(f"  Cluster {cluster_id:2d}: avg={avg_distance:.3f}, min={min_distance:.3f}, max={max_distance:.3f}")
        
        print(f"\nCluster Separation Score (higher = better separated):")
        min_inter_cluster_dist = float('inf')
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                dist = np.sqrt(((cluster_centroids[i] - cluster_centroids[j]) ** 2).sum())
                if dist < min_inter_cluster_dist:
                    min_inter_cluster_dist = dist
        
        max_intra_cluster_dist = 0
        for cluster_id in sorted(np.unique(self.clusters)):
            cluster_mask = self.clusters == cluster_id
            cluster_pca = self.X_pca[cluster_mask]
            centroid = cluster_centroids[cluster_id]
            distances_to_centroid = np.sqrt(((cluster_pca - centroid) ** 2).sum(axis=1))
            max_dist = distances_to_centroid.max()
            if max_dist > max_intra_cluster_dist:
                max_intra_cluster_dist = max_dist
        
        separation_score = min_inter_cluster_dist / max_intra_cluster_dist if max_intra_cluster_dist > 0 else 0
        print(f"  Minimum inter-cluster distance: {min_inter_cluster_dist:.3f}")
        print(f"  Maximum intra-cluster distance: {max_intra_cluster_dist:.3f}")
        print(f"  Separation ratio: {separation_score:.3f} (higher is better)")
        
        return None

"""
Same as NBADataLoader but uses cosine similarity instead of Euclidean distance.
"""
class NBADataLoaderCosine:
    
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
        return self.df
    
    def clean_data(self, min_minutes=200):
        if self.df is None:
            print("load first")
            return None

        required_minutes = self.df['mp'] >= min_minutes
        filtered_df = self.df[required_minutes]
        self.df_clean = filtered_df.copy()
        
        missing_before = self.df_clean.isnull().sum()
        self.df_clean = self.df_clean.fillna(0)
        missing_after = self.df_clean.isnull().sum().sum()
        
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
            print(f"{stat:30s}: [{min_val:7.2f}, {max_val:7.2f}] mean={mean_val:7.2f}, std={std_val:7.2f}")
        
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        for i, feat in enumerate(self.features):
            mean_val = self.X_scaled[:, i].mean()
            std_val = self.X_scaled[:, i].std()
            print(f"{feat:30s}: mean={mean_val:7.3f}, std={std_val:7.3f}")
        
        print(f"\nNormalized shape: {self.X_scaled.shape}")
        
        return self.X_scaled
    
    def pcaAnalysis(self, n_components=None):
        if self.X_scaled is None:
            print("Error: Must normalize data first!")
            return None
        
        pca_full = PCA()
        pca_full.fit(self.X_scaled)
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
        
        print("Variance for componenets:")
        for i in range(min(5, len(cumsum_variance))):
            var_explained = pca_full.explained_variance_ratio_[i]
            cum_var = cumsum_variance[i]
            print(f"PC{i+1}: {var_explained:.1%} variance (cumulative: {cum_var:.1%})")
        
        if n_components is None:
            if cumsum_variance[1] >= 0.50:
                n_components = 2
                print(f"\nUsing 2 components ({cumsum_variance[1]:.1%} variance)")
            else:
                n_components = 3
                print(f"\nUsing 3 components ({cumsum_variance[2]:.1%} variance)")
        
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        print(f"Total variance: {self.pca.explained_variance_ratio_.sum():.1%}")
        print(f"PCA data: {self.X_pca.shape}")
        
        return self.X_pca, self.pca
    
    def find_optimal_k(self, max_k=15):
        if self.X_pca is None:
            print("do PCA first")
            return None      
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            X_normalized = self.X_pca / np.linalg.norm(self.X_pca, axis=1, keepdims=True)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_normalized)
            silhouette_avg = silhouette_score(X_normalized, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        print(f"\nSilhouette scores (using cosine distance)")
        for k, score in zip(k_range, silhouette_scores):
            marker = " <-- OPTIMAL" if k == optimal_k else ""
            print(f"  k={k}: {score:.3f}{marker}")
        
        return optimal_k, silhouette_scores
    
    def cluster_players(self, n_clusters):
        if self.X_pca is None:
            print("do pca first")
            return None

        X_normalized = self.X_pca / np.linalg.norm(self.X_pca, axis=1, keepdims=True)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = self.kmeans.fit_predict(X_normalized)
        
        self.df_clean['cluster'] = self.clusters
        
        silhouette_avg = silhouette_score(X_normalized, self.clusters)
        print(f"Silhouette cosine score: {silhouette_avg:.3f}")
        
        cluster_counts = pd.Series(self.clusters).value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            print(f"Cluster {cluster_id}: {count} players")
        
        return self.clusters
    
    def analyze_clusters(self):
        if self.clusters is None:
            print("cluster first")
            return None
        
        for cluster_id in sorted(self.df_clean['cluster'].unique()):
            cluster_players = self.df_clean[self.df_clean['cluster'] == cluster_id]
            
            print(f"Cluster {cluster_id} ({len(cluster_players)} players):")
            print("-" * 60)
            
            avg_stats = cluster_players[self.features].mean()
            top_stats = avg_stats.nlargest(5)
            print("Top 5 average stats:")
            for stat, val in top_stats.items():
                print(f"{stat:30s}: {val:8.2f}")
            
            print()
    
    def visualize_clusters(self, save_plot=True):
        if self.X_pca is None or self.clusters is None:
            print("do pca and cluster first")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        projections = [
            (0, 1, 'PC1', 'PC2'),
            (0, 2, 'PC1', 'PC3'),
            (1, 2, 'PC2', 'PC3')
        ]
        
        var_explained = self.pca.explained_variance_ratio_
        
        for idx, (i, j, label_i, label_j) in enumerate(projections):
            ax = axes[idx]
            
            scatter = ax.scatter(
                self.X_pca[:, i], 
                self.X_pca[:, j], 
                c=self.clusters, 
                cmap='tab20', 
                alpha=0.6, 
                s=10
            )
            
            ax.set_xlabel(f'{label_i} ({var_explained[i]:.1%} variance)', fontsize=11)
            ax.set_ylabel(f'{label_j} ({var_explained[j]:.1%} variance)', fontsize=11)
            ax.set_title(f'{label_i} vs {label_j}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.suptitle('NBA Players Clustered in PCA Space (Cosine Similarity)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('nba_clusters_visualization_cosine.png', dpi=300, bbox_inches='tight')
            print("saved to 'nba_clusters_visualization_cosine.png'")
        
        plt.show()
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
        
        scatter2 = ax2.scatter(
            self.X_pca[:, 0], 
            self.X_pca[:, 1], 
            c=self.clusters, 
            cmap='tab20', 
            alpha=0.6, 
            s=15
        )
        
        ax2.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)', fontsize=12)
        ax2.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)', fontsize=12)
        ax2.set_title('NBA Players Clustered in 2D PCA Space - Cosine Similarity (PC1 vs PC2)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
        
        if save_plot:
            plt.savefig('nba_clusters_pc1_pc2_cosine.png', dpi=300, bbox_inches='tight')
            print("saved 'nba_clusters_pc1_pc2_cosine.png'")
        
        plt.show()
    
    def visualize_clusters_3d(self, save_plot=True):
        if self.X_pca is None or self.clusters is None:
            print("do pca and cluster first")
            return None
        
        if self.X_pca.shape[1] < 3:
            print("Need at least 3 PCA components for 3D visualization")
            return None
        
        var_explained = self.pca.explained_variance_ratio_
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            self.X_pca[:, 0],
            self.X_pca[:, 1],
            self.X_pca[:, 2],
            c=self.clusters,
            cmap='tab20',
            alpha=0.6,
            s=20
        )
        
        ax.set_xlabel(f'PC1 ({var_explained[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({var_explained[1]:.1%} variance)', fontsize=12)
        ax.set_zlabel(f'PC3 ({var_explained[2]:.1%} variance)', fontsize=12)
        ax.set_title('NBA Players Clustered in 3D PCA Space (Cosine Similarity)', fontsize=14, fontweight='bold', pad=20)
        
        plt.colorbar(scatter, ax=ax, label='Cluster', shrink=0.8)
        
        if save_plot:
            plt.savefig('nba_clusters_3d_cosine.png', dpi=300, bbox_inches='tight')
            print("Saved 3D visualization to 'nba_clusters_3d_cosine.png'")
        
        plt.show()
        
        print("3D visualization complete!")
    
    def visualize_cluster_centroids_radar(self, save_plot=True, n_features=8):
        if self.clusters is None:
            print("cluster")
            return None
        
        cluster_centroids = {}
        for cluster_id in sorted(self.df_clean['cluster'].unique()):
            cluster_data = self.df_clean[self.df_clean['cluster'] == cluster_id]
            centroids = cluster_data[self.features].mean()
            cluster_centroids[cluster_id] = centroids
        
        feature_variance = {}
        for feat in self.features:
            values = []
            for c in cluster_centroids.keys():
                values.append(cluster_centroids[c][feat])
            feature_variance[feat] = np.var(values)
        
        def get_variance(item):
            return item[1]
        sorted_items = sorted(feature_variance.items(), key=get_variance, reverse=True)
        top_features = sorted_items[:n_features]
        top_feature_names = []
        for feat in top_features:
            top_feature_names.append(feat[0])
        
        print(f"Using top {n_features} most variable features: {top_feature_names}")
        
        normalized_centroids = {}
        for cluster_id in cluster_centroids.keys():
            normalized = {}
            for feat in top_feature_names:
                all_values = []
                for c in cluster_centroids.keys():
                    all_values.append(cluster_centroids[c][feat])
                min_val = min(all_values)
                max_val = max(all_values)
                if max_val != min_val:
                    normalized[feat] = (cluster_centroids[cluster_id][feat] - min_val) / (max_val - min_val)
                else:
                    normalized[feat] = 0.5
            normalized_centroids[cluster_id] = normalized
        
        n_clusters = len(cluster_centroids)
        n_cols = 5
        n_rows = (n_clusters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows), subplot_kw=dict(projection='polar'))
        if n_clusters > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        N = len(top_feature_names)
        angles = []
        for n in range(N):
            angles.append(n / float(N) * 2 * np.pi)
        angles.append(angles[0])
        
        for idx, cluster_id in enumerate(sorted(cluster_centroids.keys())):
            ax = axes[idx]
            
            values = []
            for feat in top_feature_names:
                values.append(normalized_centroids[cluster_id][feat])
            values.append(values[0])
            
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}')
            ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            short_names = []
            for name in top_feature_names:
                short_name = name.replace('_per_36_min', '/36').replace('percent_', '%').replace('_', ' ')[:15]
                short_names.append(short_name)
            ax.set_xticklabels(short_names, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
            ax.grid(True)
            
            cluster_size = len(self.df_clean[self.df_clean['cluster'] == cluster_id])
            ax.set_title(f'Cluster {cluster_id}\n({cluster_size} players)', 
                        fontsize=10, fontweight='bold', pad=20)
        
        for idx in range(n_clusters, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Cluster Centroids - Radar Charts (Cosine Similarity)\n(Normalized feature values)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('nba_cluster_centroids_radar_cosine.png', dpi=300, bbox_inches='tight')
            print("Saved radar charts to 'nba_cluster_centroids_radar_cosine.png'")
        
        plt.show()
        
        fig2, ax2 = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        
        for idx, cluster_id in enumerate(sorted(cluster_centroids.keys())):
            values = []
            for feat in top_feature_names:
                values.append(normalized_centroids[cluster_id][feat])
            values.append(values[0])
            
            ax2.plot(angles, values, 'o-', linewidth=1.5, label=f'Cluster {cluster_id}', 
                    color=colors[idx], alpha=0.7)
            ax2.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax2.set_xticks(angles[:-1])
        short_names = []
        for name in top_feature_names:
            short_name = name.replace('_per_36_min', '/36').replace('percent_', '%').replace('_', ' ')[:20]
            short_names.append(short_name)
        ax2.set_xticklabels(short_names, fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax2.grid(True)
        ax2.set_title('All Cluster Centroids Comparison (Cosine Similarity)\n(Normalized feature values)', 
                     fontsize=14, fontweight='bold', pad=30)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        
        if save_plot:
            plt.savefig('nba_cluster_centroids_comparison_cosine.png', dpi=300, bbox_inches='tight')
            print("Saved comparison chart to 'nba_cluster_centroids_comparison_cosine.png'")
        
        plt.show()
        
        print("Radar chart visualization complete!")
    
    def find_similar_players(self, player_name, n_similar=10, method='cluster'):
        if self.X_pca is None or self.clusters is None:
            print("cluster first")
            return None
        
        print(f"\n--- Finding Similar Players to '{player_name}' (Cosine Similarity) ---")
        
        player_mask = self.df_clean['player'].str.contains(player_name, case=False, na=False)
        matching_players = self.df_clean[player_mask]
        
        if len(matching_players) == 0:
            print(f"Player '{player_name}' not found!")
            print("Try searching with a partial name (e.g., 'LeBron' instead of full name)")
            return None
        
        if len(matching_players) > 1:
            print(f"Multiple players found. Using first match:")
            for idx, row in matching_players.iterrows():
                print(f"  - {row['player']} ({row['team']}, {row['pos']})")
            print(f"\nUsing: {matching_players.iloc[0]['player']}")
        
        player_idx = matching_players.index[0]
        player_data = matching_players.iloc[0]
        player_pca = self.X_pca[player_idx]
        player_cluster = player_data['cluster']
        
        print(f"\nPlayer: {player_data['player']}")
        print(f"Team: {player_data['team']}, Position: {player_data['pos']}")
        print(f"Cluster: {player_cluster}")
        print(f"PCA coordinates: [{player_pca[0]:.2f}, {player_pca[1]:.2f}, {player_pca[2]:.2f}]")
        
        results = {}
        
        if method in ['cluster', 'both']:
            cluster_players = self.df_clean[self.df_clean['cluster'] == player_cluster].copy()
            cluster_players = cluster_players[cluster_players.index != player_idx]
            
            if len(cluster_players) > 0:
                cluster_indices = cluster_players.index
                cluster_pca = self.X_pca[cluster_indices]
                
                player_pca_norm = player_pca.reshape(1, -1)
                cluster_pca_norm = cluster_pca
                similarities = cosine_similarity(player_pca_norm, cluster_pca_norm)[0]
                
                cluster_players['similarity'] = similarities
                cluster_players = cluster_players.sort_values('similarity', ascending=False)
                
                results['cluster'] = cluster_players.head(n_similar)
        
        if method in ['distance', 'both']:
            player_pca_norm = player_pca.reshape(1, -1)
            all_pca_norm = self.X_pca
            similarities = cosine_similarity(player_pca_norm, all_pca_norm)[0]
            
            sorted_indices = np.argsort(similarities)[::-1]
            similar_indices = sorted_indices[1:n_similar+1]
            similar_players = self.df_clean.iloc[similar_indices].copy()
            similar_players['similarity'] = similarities[similar_indices]
            
            results['distance'] = similar_players
        
        if method == 'cluster' or method == 'both':
            print(f"\n{'='*70}")
            print(f"Players in SAME CLUSTER ({player_cluster}) - Cosine Similarity:")
            print(f"{'='*70}")
            if len(results['cluster']) > 0:
                for i, (idx, row) in enumerate(results['cluster'].iterrows(), 1):
                    print(f"{i:2d}. {row['player']:25s} ({row['team']:3s}, {row['pos']:2s}) "
                          f"[Similarity: {row['similarity']:.3f}]")
            else:
                print("No other players in this cluster.")
        
        if method == 'distance' or method == 'both':
            print(f"\n{'='*70}")
            print(f"Most SIMILAR PLAYERS (by cosine similarity):")
            print(f"{'='*70}")
            for i, (idx, row) in enumerate(results['distance'].iterrows(), 1):
                cluster_info = f"Cluster {int(row['cluster'])}"
                print(f"{i:2d}. {row['player']:25s} ({row['team']:3s}, {row['pos']:2s}) "
                      f"[{cluster_info:12s}, Similarity: {row['similarity']:.3f}]")
        
        return results

"""
Uses dual PCA to separate offense and defense stats.
"""
class NBADataLoaderDualPCA:

    def __init__(self, csv_path='players.csv'):
        self.csv_path = csv_path
        self.df = None
        self.df_clean = None
        self.features_offense = None
        self.features_defense = None
        self.features = None
        self.X = None
        self.X_scaled = None
        self.X_scaled_offense = None
        self.X_scaled_defense = None
        self.scaler_offense = None
        self.scaler_defense = None
        self.pca_offense = None
        self.pca_defense = None
        self.X_pca_offense = None
        self.X_pca_defense = None
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
        
        self.df_clean = self.df_clean.fillna(0)
        missing_after = self.df_clean.isnull().sum().sum()
        print(f"\nMissing values after cleaning: {missing_after}")
        
        return self.df_clean
    
    def stats_data(self):
        if self.df_clean is None:
            print("Error: Must clean data first!")
            return None

        offense_features = [
            'pts_per_36_min',
            'ast_per_36_min',
            'e_fg_percent',
            'x3p_percent',
            'avg_dist_fga',
            'percent_fga_from_x0_3_range',
            'percent_fga_from_x3p_range',
            'percent_dunks_of_fga',
            'points_generated_by_assists',
            'tov_per_36_min',
            'net_plus_minus_per_100_poss'
        ]
        
        defense_features = [
            'trb_per_36_min',
            'stl_per_36_min',
            'blk_per_36_min',
            'pf_per_36_min'
        ]
        
        available_offense = []
        for f in offense_features:
            if f in self.df_clean.columns:
                available_offense.append(f)
        
        available_defense = []
        for f in defense_features:
            if f in self.df_clean.columns:
                available_defense.append(f)
        
        missing_offense = []
        for f in offense_features:
            if f not in self.df_clean.columns:
                missing_offense.append(f)
        
        missing_defense = []
        for f in defense_features:
            if f not in self.df_clean.columns:
                missing_defense.append(f)
        
        if missing_offense:
            print(f"Offense features not found: {missing_offense}")
        if missing_defense:
            print(f"Defense features not found: {missing_defense}")
        
        self.features_offense = available_offense
        self.features_defense = available_defense
        self.features = self.features_offense + self.features_defense
        
        print(f"\n--- Feature Separation ---")
        print(f"Offensive/Shooting features ({len(self.features_offense)}):")
        for i, stat in enumerate(self.features_offense, 1):
            print(f"  {i:2d}. {stat}")
        
        print(f"\nDefensive features ({len(self.features_defense)}):")
        for i, stat in enumerate(self.features_defense, 1):
            print(f"  {i:2d}. {stat}")
        
        self.X_offense = self.df_clean[self.features_offense].values
        self.X_defense = self.df_clean[self.features_defense].values
        self.X = self.df_clean[self.features].values
        
        print(f"\nFeature matrix shapes:")
        print(f"  Offense: {self.X_offense.shape}")
        print(f"  Defense: {self.X_defense.shape}")
        print(f"  Combined: {self.X.shape}")
        
        return self.features_offense, self.features_defense, self.X
    
    def normalize_data(self):
        if self.X_offense is None or self.X_defense is None:
            print("Select features first")
            return None
        
        print(f"\n--- Normalizing Data (Separate for Offense and Defense) ---")
        
        print("\nOffense features (before normalization):")
        for i, stat in enumerate(self.features_offense):
            min_val = self.X_offense[:, i].min()
            max_val = self.X_offense[:, i].max()
            mean_val = self.X_offense[:, i].mean()
            std_val = self.X_offense[:, i].std()
            print(f"  {stat:30s}: [{min_val:7.2f}, {max_val:7.2f}] mean={mean_val:7.2f}, std={std_val:7.2f}")
        
        self.scaler_offense = StandardScaler()
        self.X_scaled_offense = self.scaler_offense.fit_transform(self.X_offense)
        
        print("\nDefense features (before normalization):")
        for i, stat in enumerate(self.features_defense):
            min_val = self.X_defense[:, i].min()
            max_val = self.X_defense[:, i].max()
            mean_val = self.X_defense[:, i].mean()
            std_val = self.X_defense[:, i].std()
            print(f"  {stat:30s}: [{min_val:7.2f}, {max_val:7.2f}] mean={mean_val:7.2f}, std={std_val:7.2f}")
        
        self.scaler_defense = StandardScaler()
        self.X_scaled_defense = self.scaler_defense.fit_transform(self.X_defense)
        
        self.X_scaled = np.hstack([self.X_scaled_offense, self.X_scaled_defense])
        
        print(f"\nNormalized shapes:")
        print(f"  Offense: {self.X_scaled_offense.shape}")
        print(f"  Defense: {self.X_scaled_defense.shape}")
        print(f"  Combined: {self.X_scaled.shape}")
        
        return self.X_scaled_offense, self.X_scaled_defense
    
    def pcaAnalysis(self, n_components_offense=2, n_components_defense=1):
        if self.X_scaled_offense is None or self.X_scaled_defense is None:
            print("Error: Must normalize data first!")
            return None
        
        print(f"\n--- Dual PCA: Separate PCA for Offense and Defense ---")
        
        print("\n1. PCA on Offensive/Shooting Features:")
        pca_offense_full = PCA()
        pca_offense_full.fit(self.X_scaled_offense)
        cumsum_offense = np.cumsum(pca_offense_full.explained_variance_ratio_)
        
        print("Variance explained by offense components:")
        for i in range(min(5, len(cumsum_offense))):
            var_explained = pca_offense_full.explained_variance_ratio_[i]
            cum_var = cumsum_offense[i]
            print(f"  PC{i+1}: {var_explained:.1%} variance (cumulative: {cum_var:.1%})")
        
        self.pca_offense = PCA(n_components=n_components_offense)
        self.X_pca_offense = self.pca_offense.fit_transform(self.X_scaled_offense)
        print(f"  Using {n_components_offense} offense components ({self.pca_offense.explained_variance_ratio_.sum():.1%} variance)")
        
        print("\n2. PCA on Defensive Features:")
        pca_defense_full = PCA()
        pca_defense_full.fit(self.X_scaled_defense)
        cumsum_defense = np.cumsum(pca_defense_full.explained_variance_ratio_)
        
        print("Variance explained by defense components:")
        for i in range(min(5, len(cumsum_defense))):
            var_explained = pca_defense_full.explained_variance_ratio_[i]
            cum_var = cumsum_defense[i]
            print(f"  PC{i+1}: {var_explained:.1%} variance (cumulative: {cum_var:.1%})")
        
        self.pca_defense = PCA(n_components=n_components_defense)
        self.X_pca_defense = self.pca_defense.fit_transform(self.X_scaled_defense)
        print(f"  Using {n_components_defense} defense components ({self.pca_defense.explained_variance_ratio_.sum():.1%} variance)")
        
        self.X_pca = np.hstack([self.X_pca_offense, self.X_pca_defense])
        
        total_components = n_components_offense + n_components_defense
        print(f"\n3. Concatenated PCA:")
        print(f"  Combined shape: {self.X_pca.shape} (players × {total_components} components)")
        print(f"  Components: {n_components_offense} offense + {n_components_defense} defense = {total_components} total")
        
        return self.X_pca, (self.pca_offense, self.pca_defense)
    
    def find_optimal_k(self, max_k=15):
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
        
        print(f"\nSilhouette scores (Dual PCA)")
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
    
    def analyze_clusters(self):
        if self.clusters is None:
            print("Error: Must cluster players first!")
            return None
        
        print(f"\n--- Cluster Analysis (Dual PCA) ---")
        print("Average statistics for each cluster:\n")
        
        for cluster_id in sorted(self.df_clean['cluster'].unique()):
            cluster_players = self.df_clean[self.df_clean['cluster'] == cluster_id]
            
            print(f"Cluster {cluster_id} ({len(cluster_players)} players):")
            print("-" * 60)
            
            avg_stats = cluster_players[self.features].mean()
            top_stats = avg_stats.nlargest(5)
            print("Top 5 average stats:")
            for stat, val in top_stats.items():
                print(f"  {stat:30s}: {val:8.2f}")
            
            print()
    
    def visualize_clusters(self, save_plot=True):
        if self.X_pca is None or self.clusters is None:
            print("do pca and cluster first")
            return None
        
        n_components = self.X_pca.shape[1]
        
        if n_components >= 3:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            projections = [
                (0, 1, 'Offense PC1', 'Offense PC2'),
                (0, 2, 'Offense PC1', 'Defense PC1'),
                (1, 2, 'Offense PC2', 'Defense PC1')
            ]
            
            for idx, (i, j, label_i, label_j) in enumerate(projections):
                ax = axes[idx]
                
                scatter = ax.scatter(
                    self.X_pca[:, i], 
                    self.X_pca[:, j], 
                    c=self.clusters, 
                    cmap='tab20', 
                    alpha=0.6, 
                    s=10
                )
                
                ax.set_xlabel(f'{label_i}', fontsize=11)
                ax.set_ylabel(f'{label_j}', fontsize=11)
                ax.set_title(f'{label_i} vs {label_j}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Cluster')
            
            plt.suptitle('NBA Players Clustered - Dual PCA (Offense + Defense)', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            if save_plot:
                plt.savefig('nba_clusters_visualization_dual_pca.png', dpi=300, bbox_inches='tight')
                print("Saved visualization to 'nba_clusters_visualization_dual_pca.png'")
            
            plt.show()
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
        
        scatter2 = ax2.scatter(
            self.X_pca[:, 0], 
            self.X_pca[:, 1], 
            c=self.clusters, 
            cmap='tab20', 
            alpha=0.6, 
            s=15
        )
        
        ax2.set_xlabel('Offense PC1', fontsize=12)
        ax2.set_ylabel('Offense PC2', fontsize=12)
        ax2.set_title('NBA Players Clustered - Dual PCA (Offense PC1 vs Offense PC2)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
        
        if save_plot:
            plt.savefig('nba_clusters_dual_pca_main.png', dpi=300, bbox_inches='tight')
            print("Saved main visualization to 'nba_clusters_dual_pca_main.png'")
        
        plt.show()
        
        print("Visualization complete!")
    
    def visualize_clusters_3d(self, save_plot=True):
        if self.X_pca is None or self.clusters is None:
            print("do pca and cluster first")
            return None
        
        if self.X_pca.shape[1] < 3:
            print("Need at least 3 PCA components for 3D visualization")
            return None
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            self.X_pca[:, 0],
            self.X_pca[:, 1],
            self.X_pca[:, 2],
            c=self.clusters,
            cmap='tab20',
            alpha=0.6,
            s=20
        )
        
        ax.set_xlabel('Offense PC1', fontsize=12)
        ax.set_ylabel('Offense PC2', fontsize=12)
        ax.set_zlabel('Defense PC1', fontsize=12)
        ax.set_title('NBA Players Clustered in 3D PCA Space (Dual PCA)', fontsize=14, fontweight='bold', pad=20)
        
        plt.colorbar(scatter, ax=ax, label='Cluster', shrink=0.8)
        
        if save_plot:
            plt.savefig('nba_clusters_3d_dual_pca.png', dpi=300, bbox_inches='tight')
            print("Saved 3D visualization to 'nba_clusters_3d_dual_pca.png'")
        
        plt.show()
        
        print("3D visualization complete!")
    
    def visualize_cluster_centroids_radar(self, save_plot=True, n_features=8):
        if self.clusters is None:
            print("Error: Must cluster players first!")
            return None
        
        print(f"\n--- Creating Radar Charts for Cluster Centroids (Dual PCA) ---")
        
        cluster_centroids = {}
        for cluster_id in sorted(self.df_clean['cluster'].unique()):
            cluster_data = self.df_clean[self.df_clean['cluster'] == cluster_id]
            centroids = cluster_data[self.features].mean()
            cluster_centroids[cluster_id] = centroids
        
        feature_variance = {}
        for feat in self.features:
            values = []
            for c in cluster_centroids.keys():
                values.append(cluster_centroids[c][feat])
            feature_variance[feat] = np.var(values)
        
        def get_variance(item):
            return item[1]
        sorted_items = sorted(feature_variance.items(), key=get_variance, reverse=True)
        top_features = sorted_items[:n_features]
        top_feature_names = []
        for feat in top_features:
            top_feature_names.append(feat[0])
        
        print(f"Using top {n_features} most variable features: {top_feature_names}")
        
        normalized_centroids = {}
        for cluster_id in cluster_centroids.keys():
            normalized = {}
            for feat in top_feature_names:
                all_values = []
                for c in cluster_centroids.keys():
                    all_values.append(cluster_centroids[c][feat])
                min_val = min(all_values)
                max_val = max(all_values)
                if max_val != min_val:
                    normalized[feat] = (cluster_centroids[cluster_id][feat] - min_val) / (max_val - min_val)
                else:
                    normalized[feat] = 0.5
            normalized_centroids[cluster_id] = normalized
        
        n_clusters = len(cluster_centroids)
        n_cols = 5
        n_rows = (n_clusters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows), subplot_kw=dict(projection='polar'))
        if n_clusters > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        N = len(top_feature_names)
        angles = []
        for n in range(N):
            angles.append(n / float(N) * 2 * np.pi)
        angles.append(angles[0])
        
        for idx, cluster_id in enumerate(sorted(cluster_centroids.keys())):
            ax = axes[idx]
            
            values = []
            for feat in top_feature_names:
                values.append(normalized_centroids[cluster_id][feat])
            values.append(values[0])
            
            ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {cluster_id}')
            ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            short_names = []
            for name in top_feature_names:
                short_name = name.replace('_per_36_min', '/36').replace('percent_', '%').replace('_', ' ')[:15]
                short_names.append(short_name)
            ax.set_xticklabels(short_names, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
            ax.grid(True)
            
            cluster_size = len(self.df_clean[self.df_clean['cluster'] == cluster_id])
            ax.set_title(f'Cluster {cluster_id}\n({cluster_size} players)', 
                        fontsize=10, fontweight='bold', pad=20)
        
        for idx in range(n_clusters, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Cluster Centroids - Radar Charts (Dual PCA)\n(Normalized feature values)', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('nba_cluster_centroids_radar_dual_pca.png', dpi=300, bbox_inches='tight')
            print("Saved radar charts to 'nba_cluster_centroids_radar_dual_pca.png'")
        
        plt.show()
        
        fig2, ax2 = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        
        for idx, cluster_id in enumerate(sorted(cluster_centroids.keys())):
            values = []
            for feat in top_feature_names:
                values.append(normalized_centroids[cluster_id][feat])
            values.append(values[0])
            
            ax2.plot(angles, values, 'o-', linewidth=1.5, label=f'Cluster {cluster_id}', 
                    color=colors[idx], alpha=0.7)
            ax2.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax2.set_xticks(angles[:-1])
        short_names = []
        for name in top_feature_names:
            short_name = name.replace('_per_36_min', '/36').replace('percent_', '%').replace('_', ' ')[:20]
            short_names.append(short_name)
        ax2.set_xticklabels(short_names, fontsize=10)
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax2.grid(True)
        ax2.set_title('All Cluster Centroids Comparison (Dual PCA)\n(Normalized feature values)', 
                     fontsize=14, fontweight='bold', pad=30)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
        
        if save_plot:
            plt.savefig('nba_cluster_centroids_comparison_dual_pca.png', dpi=300, bbox_inches='tight')
            print("Saved comparison chart to 'nba_cluster_centroids_comparison_dual_pca.png'")
        
        plt.show()
        
        print("Radar chart visualization complete!")
    
    def find_similar_players(self, player_name, n_similar=10, method='cluster'):
        if self.X_pca is None or self.clusters is None:
            print("cluster first")
            return None
        
        print(f"\n--- Finding Similar Players to '{player_name}' (Dual PCA) ---")
        
        player_mask = self.df_clean['player'].str.contains(player_name, case=False, na=False)
        matching_players = self.df_clean[player_mask]
        
        if len(matching_players) == 0:
            print(f"Player '{player_name}' not found!")
            return None
        
        if len(matching_players) > 1:
            print(f"Multiple players found. Using first match:")
            for idx, row in matching_players.iterrows():
                print(f"  - {row['player']} ({row['team']}, {row['pos']})")
            print(f"\nUsing: {matching_players.iloc[0]['player']}")
        
        player_idx = matching_players.index[0]
        player_data = matching_players.iloc[0]
        player_pca = self.X_pca[player_idx]
        player_cluster = player_data['cluster']
        
        print(f"\nPlayer: {player_data['player']}")
        print(f"Team: {player_data['team']}, Position: {player_data['pos']}")
        print(f"Cluster: {player_cluster}")
        print(f"PCA coordinates: {player_pca}")
        
        results = {}
        
        if method in ['cluster', 'both']:
            cluster_players = self.df_clean[self.df_clean['cluster'] == player_cluster].copy()
            cluster_players = cluster_players[cluster_players.index != player_idx]
            
            if len(cluster_players) > 0:
                cluster_indices = cluster_players.index
                cluster_pca = self.X_pca[cluster_indices]
                distances = np.sqrt(((cluster_pca - player_pca) ** 2).sum(axis=1))
                cluster_players['distance'] = distances
                cluster_players = cluster_players.sort_values('distance')
                results['cluster'] = cluster_players.head(n_similar)
        
        if method in ['distance', 'both']:
            distances = np.sqrt(((self.X_pca - player_pca) ** 2).sum(axis=1))
            similar_indices = np.argsort(distances)[1:n_similar+1]
            similar_players = self.df_clean.iloc[similar_indices].copy()
            similar_players['distance'] = distances[similar_indices]
            results['distance'] = similar_players
        
        if method == 'cluster' or method == 'both':
            print(f"\n{'='*70}")
            print(f"Players in SAME CLUSTER ({player_cluster}):")
            print(f"{'='*70}")
            if len(results['cluster']) > 0:
                for i, (idx, row) in enumerate(results['cluster'].iterrows(), 1):
                    print(f"{i:2d}. {row['player']:25s} ({row['team']:3s}, {row['pos']:2s}) "
                          f"[Distance: {row['distance']:.3f}]")
            else:
                print("No other players in this cluster.")
        
        if method == 'distance' or method == 'both':
            print(f"\n{'='*70}")
            print(f"Most SIMILAR PLAYERS (by PCA distance):")
            print(f"{'='*70}")
            for i, (idx, row) in enumerate(results['distance'].iterrows(), 1):
                cluster_info = f"Cluster {int(row['cluster'])}"
                print(f"{i:2d}. {row['player']:25s} ({row['team']:3s}, {row['pos']:2s}) "
                      f"[{cluster_info:12s}, Distance: {row['distance']:.3f}]")
        
        return results


if __name__ == "__main__":
    loader = NBADataLoader(csv_path='players.csv')
    df = loader.load_data()
    print("First few rows of the data:")
    print(df.head())
    print("\nmissing if any:")
    print(df.isnull().sum())
    
    df_clean = loader.clean_data(min_minutes=200)
    print(f"\nCleaned data shape: {df_clean.shape}")
    print(df_clean[['player', 'team', 'pos', 'mp']].head(10))
    
    features, X = loader.stats_data()
    print(f"\nSample of feature values (first 3 players, first 5 features):")
    print(X[:3, :5])
    
    X_scaled = loader.normalize_data()
    print(f"\nSample of normalized values (first 3 players, first 5 features):")
    print(X_scaled[:3, :5])
    
    X_pca, pca = loader.pcaAnalysis(n_components=3)
    print(f"\nSample of PCA values (first 3 players, all components):")
    print(X_pca[:3, :])
    
    optimal_k, scores = loader.find_optimal_k(max_k=15)
    use_k = 15
    
    clusters = loader.cluster_players(n_clusters=use_k)
    loader.analyze_clusters()
    loader.show_cluster_similarities()
    
    print(f"\nSample players from each cluster:")
    for cluster_id in sorted(loader.df_clean['cluster'].unique()):
        cluster_players = loader.df_clean[loader.df_clean['cluster'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_players)} players):")
        print(cluster_players[['player', 'team', 'pos']].head(5).to_string(index=False))
    
    print("\n" + "="*70)
    print("EXAMPLE: Finding similar players")
    print("="*70)
    loader.find_similar_players("LeBron James", n_similar=10, method='both')
    loader.find_similar_players("Stephen Curry", n_similar=10, method='both')
    loader.find_similar_players("Giannis", n_similar=10, method='both')
    
    
    #USING COSINE SIMILARITY
    print("\n" + "="*70)
    print("COMPARISON: Using COSINE SIMILARITY")
    print("="*70)
    
    loader_cosine = NBADataLoaderCosine(csv_path='players.csv')
    df_cosine = loader_cosine.load_data()
    df_clean_cosine = loader_cosine.clean_data(min_minutes=200)
    features_cosine, X_cosine = loader_cosine.stats_data()
    X_scaled_cosine = loader_cosine.normalize_data()
    X_pca_cosine, pca_cosine = loader_cosine.pcaAnalysis(n_components=3)
    
    optimal_k_cosine, scores_cosine = loader_cosine.find_optimal_k(max_k=15)
    use_k_cosine = 15
    clusters_cosine = loader_cosine.cluster_players(n_clusters=use_k_cosine)
    loader_cosine.analyze_clusters()
    
    # Visualize
    loader_cosine.visualize_clusters(save_plot=True)
    loader_cosine.visualize_clusters_3d(save_plot=True)
    loader_cosine.visualize_cluster_centroids_radar(save_plot=True, n_features=8)
    
    print("\n" + "="*70)
    print("Finding similar players using COSINE SIMILARITY")
    print("="*70)
    loader_cosine.find_similar_players("LeBron James", n_similar=10, method='both')
    loader_cosine.find_similar_players("Stephen Curry", n_similar=10, method='both')
    loader_cosine.find_similar_players("Giannis", n_similar=10, method='both')
    
    # ============================================================================
    # COMPARISON: Dual PCA - Separate PCA for Offense and Defense
    # ============================================================================
    print("\n" + "="*70)
    print("COMPARISON: Using DUAL PCA (Offense + Defense separately)")
    print("="*70)
    
    loader_dual = NBADataLoaderDualPCA(csv_path='players.csv')
    df_dual = loader_dual.load_data()
    df_clean_dual = loader_dual.clean_data(min_minutes=200)
    features_offense_dual, features_defense_dual, X_dual = loader_dual.stats_data()
    X_scaled_dual = loader_dual.normalize_data()
    
    # Perform dual PCA: 2 components for offense, 1 for defense
    X_pca_dual, pca_dual = loader_dual.pcaAnalysis(n_components_offense=2, n_components_defense=1)
    
    optimal_k_dual, scores_dual = loader_dual.find_optimal_k(max_k=15)
    use_k_dual = 15
    clusters_dual = loader_dual.cluster_players(n_clusters=use_k_dual)
    loader_dual.analyze_clusters()
    
    # Visualize
    loader_dual.visualize_clusters(save_plot=True)
    loader_dual.visualize_clusters_3d(save_plot=True)
    loader_dual.visualize_cluster_centroids_radar(save_plot=True, n_features=8)
    
    # Find similar players using dual PCA
    print("\n" + "="*70)
    print("Finding similar players using DUAL PCA")
    print("="*70)
    loader_dual.find_similar_players("LeBron James", n_similar=10, method='both')
    loader_dual.find_similar_players("Stephen Curry", n_similar=10, method='both')
    loader_dual.find_similar_players("Giannis", n_similar=10, method='both')
