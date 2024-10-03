### Imports

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from matplotlib import cm


# Loading the data
pathname = 'Data/SAheart.data.txt'
data = pd.read_csv(pathname, sep=",", header=0, index_col='row.names')


# Give an account of whether there are data issues (i.e., missing values or corrupted data) and describe them if so.
missing_data = data.isnull().sum()
missing_data[missing_data > 0]


# Include basic summary statistics of the attributes.
numeric_columns = ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea', 'obesity', 'alcohol', 'age']

print(f'The dataset has {data.shape[0]} rows and {data.shape[1]} columns.')
data[numeric_columns].describe()


# Are there issues with outliers in the data?
plt.figure(figsize=(10, 6))
boxplot = sns.boxplot(data=data[numeric_columns], 
                      color='steelblue', 
                      showcaps=True, 
                      fill=False,
                      flierprops=dict(marker='x', color='firebrick', markersize=6, markeredgecolor='firebrick'),
                      medianprops=dict(color='firebrick', linewidth=2),
                      whiskerprops=dict(color='black', linewidth=1.5, linestyle='dashed'),
                      capprops=dict(color='black', linewidth=1.5))

plt.grid(True, which='both', axis='y', color='lightgray', linestyle='--', linewidth=0.5)
plt.xticks()
plt.show()


## Violin plot 
plt.figure(figsize=(10, 6))
boxplot = sns.violinplot(data=data[numeric_columns])


# Do the attributes appear to be normally distributed?
data[numeric_columns].hist(bins=15, figsize=(15, 10), color='steelblue', edgecolor='black', grid=False)
plt.show()


# Are variables correlated?
plt.figure(figsize=(10, 8))
correlation_matrix = data[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()



### PCA
# One column in the dataframe that is not a numeric value
data['famhist'] = data['famhist'].map({'Present': 1, 'Absent': 0})

# Splitting the variables and target data
X = np.asarray(data.iloc[:, :-1].values)
y = np.asarray(data['chd'].values) 

# Only continuous variables for PCA
X_continuous = data[numeric_columns].values

# Some basic variables following the class exercises
N = len(y) #samples
M = X.shape[1] #features
C = len(np.unique(y))  #unique classes

# Normalizing the data for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_continuous)

# Centering the data for PCA
Y = X_scaled - np.ones((X_scaled.shape[0], 1)) * X_scaled.mean(axis=0) #subtracts mean value from data

# Computing singular value decomposition (SVD) of y 
U, S, Vh = svd(Y, full_matrices=False)

# Computing the amount of variance explained by the principal components
rho = (S * S) / (S * S).sum()

# Plotting variance explained per component and cumulative variance explained
fig, ax = plt.subplots(1, 1, figsize=(7, 5))

threshold = 0.9
threshold_ = 0.7

ax.plot(range(1, len(rho) + 1), rho, "x-", color='darkblue')
ax.plot(range(1, len(rho) + 1), np.cumsum(rho), "o-", color='red')
ax.plot([1, len(rho)], [threshold, threshold], "k--", color='green')
ax.plot([1, len(rho)], [threshold_, threshold_], "k--", color='black')
ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Explained")
ax.legend(["Individual", "Cumulative", "Threshold = 0.9", "Threshold = 0.7"])
ax.grid()

# Vh is the Hermitian (transpose)of the vector V - so we transpose to get V:
V = Vh.T

# Projecting the centered data onto principal component space
Z = Y @ V

# We want to use 7 principal components to explain 90% variance
Z_7 = Z[:, :7]

# Storing the new dataframe
pca_df = pd.DataFrame(Z_7, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'])
pca_df['Class'] = y

# Plot PC2 vs. PC1
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Class'], cmap='bwr', alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')

# Access legend objects automatically created from data
handles, labels = plt.gca().get_legend_handles_labels() # empty

point2 = Line2D([0], [0], label='CHD = 1', marker='o', markersize=5, linestyle='',  markerfacecolor='red', markeredgecolor='r')
point1 = Line2D([0], [0], label='CHD = 0', marker='o', markersize=5, linestyle='',  markerfacecolor='blue')

# add manual symbols to auto legend
handles.extend([point1, point2])

plt.legend(handles=handles)
plt.show()


# Grid of pairs of PCs plotted against each other
i = 0
j = 1

fig, axs = plt.subplots(7, 3, figsize=(15, 20))
pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), 
         (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]

cmap = cm.get_cmap('bwr', len(np.unique(pca_df['Class'])))

for index, (i, j) in enumerate(pairs):
    ax = axs[index // 3, index % 3]
    for c_idx, c in enumerate(np.unique(pca_df['Class'])):
        class_mask = (pca_df['Class'] == c)
        ax.scatter(pca_df[class_mask]['PC' + str(i + 1)], 
                   pca_df[class_mask]['PC' + str(j + 1)],
                   label=f'Class {c}', color=cmap(c_idx), alpha=0.6)
    
    ax.set_xlabel(f"PC{i + 1}")
    ax.set_ylabel(f"PC{j + 1}")
    ax.legend()

plt.tight_layout()
plt.show()


# Loadings of all PCs
loadings = pd.DataFrame(V[:, :7], index=numeric_columns, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'])
print(loadings)