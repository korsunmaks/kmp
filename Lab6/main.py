import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

path="Seeds.csv"
dataset =  pd.read_csv("Seeds.csv")
seed_size = dataset.iloc[:, :-1].values
seed_class = dataset.iloc[:, 7].values

num_clusters=3

# Feature normalization
scaler=MinMaxScaler()
scaler_seed_size=scaler.fit_transform(seed_size)

# K-means clustering
clusterer = KMeans(n_clusters=num_clusters)
clusterer.fit(seed_size)
labels = clusterer.labels_
metrics.silhouette_score(seed_size, labels, metric="euclidean")

# Distibution seeds bu clusters
predictions = clusterer.predict(seed_size)

# Add results to the table
dataset["cluster"] = predictions
print("Results:")
print(dataset, "\n")


# Centers of all clusters
centroids=clusterer.cluster_centers_
# print("Сoordinates of all centroids: ")
# print(centroids, "\n")


#Visualization of clustering results
def get_clustering_results_visualization(i,j):
    fig, ax = plt.subplots()
    scatter1 = ax.scatter(scaler_seed_size[:, i], scaler_seed_size[:, j],
                          c=predictions, s=15, cmap="brg")
    handles, labels = scatter1.legend_elements()
    legend1 = ax.legend(handles, labels, loc="upper right")
    ax.add_artist(legend1)
    scatter2 = ax.scatter(centroids[:, i], centroids[:, j], marker="x",
                          c="purple", s=200, linewidth=3, label="centroids")
    plt.legend(loc="lower right")
    plt.xlabel(f"{dataset.columns[i]} after scaling")
    plt.ylabel(f"{dataset.columns[j]} after scaling")
    plt.show()

#get_clustering_results_visualization(4,5)

# Amount of objects of each class in each cluster
cluster_content = dataset.groupby(["cluster", "class"]).size().unstack(fill_value=0)
cluster_content["Total"] = cluster_content.sum(axis=1)
cluster_content.loc["Total"] = cluster_content.sum()
print(tabulate(cluster_content, headers="keys", tablefmt="psql"))



# Determination of the optimal number of clusters
df = pd.DataFrame(columns=["Number of clusters", "WCSS", "Silhouette", "DB"])
for i in range(2, 11):
    clusterer_i = KMeans(n_clusters=i).fit(scaler_seed_size)
    predictions_i = clusterer_i.predict(scaler_seed_size)

    # Sum of squared distances from instances to the nearest centroid (WCSS)
    WCSS = clusterer_i.inertia_

    # Silhouette Score
    Silhouette = metrics.silhouette_score(scaler_seed_size, predictions_i)

    # Davies-Boudin Score
    DB = metrics.davies_bouldin_score(scaler_seed_size, predictions_i)

    new_row_df = pd.DataFrame([[i, WCSS, Silhouette, DB]], columns=df.columns)
    df = pd.concat([df, new_row_df], ignore_index=True)
print(tabulate(df, headers="keys", tablefmt="psql", floatfmt=".3f"))


# plt.plot(df["Number of clusters"], df["WCSS"], marker="o", linestyle="None", label="WCSS")
# plt.xlabel("Number of clusters")
# plt.ylabel("WCSS")
# plt.title("Elbow method")
# plt.legend()
# plt.show()

# plt.plot(df["Number of clusters"], df["Silhouette"], marker="o", linestyle="None", label="Silhouette")
# plt.xlabel("Number of clusters")
# plt.ylabel("Silhouette")
# plt.title("Silhouette method")
# plt.legend()
# plt.show()

plt.plot(df["Number of clusters"], df["DB"], marker="o", linestyle="None", label="DB")
plt.xlabel("Number of clusters")
plt.ylabel("DB")
plt.title("DB")
plt.legend()
plt.show()



