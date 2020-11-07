# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # IML/PRSTIM project : NMR Signal & Peak Detection
# 
# Edwin Grappin & Guillaume Tochon
# %% [markdown]
# ## Overall objective
# 
# This challenge works on nuclear magnetic resonance (NMR) observation to analyze leaves of plants. Studying leaves under RMN allows to understand the position of water within the cells.
# 
# Understanding the distribution of water in leaf cells is insightful to understand the level of senescence (the degradation due to the age) of the leaves.
# Indeed, in advanced stage of senescence, the amount of water in the vacuoles increases. Measuring the level of senescence is useful to assess the impact of environmental factors on the plant. In particular, in R&D, some scientists think about using NMR to compare the efficacy of treatments (products that are applied to the plants to protect the plants from different stresses such as drought, heat, disease...).
# 
# In this settings processing NMR signals to assess the stage of senescence of plants could be used to discover new plants treatments.
# %% [markdown]
# ![image.png](attachment:image.png)
# %% [markdown]
# ## Data description
# %% [markdown]
# Data are composed of leaf spectra (non-normalized densities) of T2 relaxation time (go read again your IMED course if you don't remember what is a T2 relaxation time), such as the one displayed below. The x-axis represents time (in ms) while the y-axis represent the intensity.
# %% [markdown]
# ![image.png](attachment:image.png)
# %% [markdown]
# The spectrum is composed of peaks of various intensities and widths, and occuring at different relaxation times. Positions and intensities of the peaks correlate with the leaf age: the older the leaf, the higher the peak intensity and the later the relaxation time. 
# 
# Leaves are labeled by rank depending on their age on the plant.
# 
# ![image.png](attachment:image.png)
# %% [markdown]
# 
# The higher the rank, the younger the leaf. In the figure below for instance, one can see the NMR response of a leaf getting older (from rank 6 to rank 3). At rank 6, there is a single peak with high intensity. Then, the T2 relaxation time of this peak is increasing in rank 5. In rank 4, the peak starts to split in two. At rank 3, the split is very clear. 
# %% [markdown]
# ![image.png](attachment:image.png)
# 
# %% [markdown]
# Information related to the number and the position of peaks in a given spectrum, as well as their intensity and width, are important information to caracterize the way a plant (and its leaves) react to environmental conditions and stress. 
# %% [markdown]
# ## Goal of the project
# The goal of this project is two-fold. First, detecting and classifying the type of peaks in the NMR spectra. Second, predicting their position. 
# For that matter, you will have the opportunity to use different machine learning and data mining methods.
# 
# Inputs are organized as pj2 files (you can open them with python as any text files) describing the spectrum (with first column being the abscisse and the second column being the intensity of the signal). 
# %% [markdown]
# ![image.png](attachment:image.png)
# %% [markdown]
# We are interested in the caracterization of 5 classes of peaks: 1, 2, 3, 4 and 5. 
# Not all classes of peaks exist in every observation. The only structural link between peaks is that peak 5 exists only if peak 4 exists.
# 
# In order to train an estimator of peaks classes and the detection, you have access to 420 NMR signals and their associated peaks in the file data_nmr_input_epita_train.csv. A row is an observation, each column is the position in ms of the peak (if it exists). Therefore, your goal is to predict on each NMR observation which peak exist and then, provided it exists, to predict its position (in ms).
# %% [markdown]
# ![image.png](attachment:image.png)
# %% [markdown]
# Data are organized in modality, replicate (observation) and rank. The file 7-3-001.pj2 is the NMR signal of the first rank of the 3 replicate of the modality 7. You have to extract and store the .zip file in the folder /app/Data/NMRBlue/LearningDataSet.
# %% [markdown]
# ![image.png](attachment:image.png)
# %% [markdown]
# ## Project organization 
# 
# Your output will be a jupyter lab notebook. The code should run (when docker container is ran) in this single jupyter notebook (but you can use other file that you import). Feel free to use markdown to explain and answer questions. Do not change container configuration. If extra installation is needed, please install directly in the notebook with `!pip install ...`. Your notebook should be self sufficient (within the docker container). 
# 
# You'll commit your results in your own branch called output/login1_login2<_login3> where _login1_ and _login2_ are your 2 EPITA logins (or 3 if you are three in the group). You will be working by groups of 2 for this project if you decide to commit your report by the end of August or by end of July if you decide to work in a group of 3.
# 
# The evaluation will be based on your ability to show that you understand machine learning concepts (such as clustering, classification) and notions related to statistics and probabilities. Good prediction performance will be valued as well, but a smart understanding of the topics and good choice of data processing and algorithms are what we are looking for. 
# 
# Due date: Last minute of July 2020 if you are a group of 3 or the last minute of August 2020 if you are a group of 2. 

# %%
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# %% [markdown]
# The first sections (Parts 1, 2, 3 and 4) are mandatory. The "optional" section is... optional. 
# 
# ## Part 1: understanding the data
# %% [markdown]
# **I.1)** Create a function that takes as argument a modality number, a replicate number and that plot a serie of spectrum. Plot the spectrum in the notebook. What can you say about the data. What are the range of values ? are the spectrums densities ? Why, why not ? Are the data in one replicate independent from each other ? 

# %%
def read_spectrum(modality=None, replicate=None, rank=None):
    if modality == None and replicate == None and rank == None:
        spectrums = {}
        for val in sorted(glob.glob('LearningDataSet/*/spectre/*'), key=lambda x: x[17]):
            with open(val) as f:
                file = []
                for line in f:
                    temp = line.split(' ')
                    if (len(temp) == 2):
                        file.append([float(i) for i in temp])
            spectrums[val] = np.moveaxis(np.array(file), 1, 0)
        return spectrums
    else:
        X = []
        Y = []
        with open(f'LearningDataSet/{modality}/spectre/{modality}-{replicate}-{rank}.pj2') as f:
            for line in f:
                temp = line.split(' ')
                if (len(temp) == 2):
                    X.append(float(temp[0]))
                    Y.append(float(temp[1]))
        return np.asarray(X), np.asarray(Y)

spectrums = read_spectrum()


# %%
def plot_spectrums(modality, replicate):
    ranks = glob.glob(f'LearningDataSet/{modality}/spectre/{modality}-{replicate}*')

    plt.figure(figsize = (15, 12))
    for i, val in enumerate(sorted(ranks)):
        sp = spectrums[val]
        plt.subplot(len(ranks)//3 + 1, 3, i+1)
        plt.plot(sp[0], sp[1], label=f'rank {i+1}')
        plt.xlabel("time (ms)")
        plt.ylabel("intensity")
        plt.legend()
    plt.show()


# %%
plot_spectrums(12, 3)

# %% [markdown]
# What are the range of values ?  
# The range of values change from one spectra to another because they are separate measurements.
# 
# Are the spectrums densities ? Why, why not ? Are the data in one replicate independent from each other ?  
# These are not densities because the area under the curve is not equal to one.
# %% [markdown]
# **I.2)** Create a normalization function to turn all spectra into density probability functions.

# %%
def normalize_all(spectrums, normalize_X):
    for key in spectrums:
        if normalize_X:
            spectrums[key][0] /= spectrums[key][0].max()
        # density probability function -> AUC=1
        auc = np.trapz(spectrums[key][1], x=spectrums[key][0])
        spectrums[key][1] = spectrums[key][1]/auc
    return spectrums


# %%
def normalize(X, Y):
    scale = X.max()
    X /= X.max()
    auc = np.trapz(Y, x=X)
    Y = Y/auc
    return X, Y, scale


# %%
plt.figure(figsize=(15, 10))

spec = spectrums['LearningDataSet/12/spectre/12-3-002.pj2']

plt.subplot(211)
plt.plot(spec[0], spec[1])
plt.xlabel("time (ms)")
plt.ylabel("intensity")
plt.title('before normalization')


spectrums = normalize_all(spectrums, True)

norm = spectrums['LearningDataSet/12/spectre/12-3-002.pj2']

plt.subplot(212)
plt.plot(norm[0], norm[1])
plt.xlabel("time (ms)")
plt.ylabel("intensity")
plt.title('after normalization')

plt.show()


# %%
print('area under the curve =', np.trapz(norm[1], x=norm[0]))

# %% [markdown]
# ## Part 2: Clustering approach and sampling to estimate the number of peaks.
# %% [markdown]
# The goal of this section is to try and differentiate the rank of the various spectra you have at hand with clustering approaches. The main issue comes from the fact that it is merely impossible to classify high dimensional data (that is, if you consider each spectrum as a vector whose dimension is the number of samples in the spectrum) without any prior dimension reduction method. So, let's try another approach, whose overall idea is the following : instead of clustering the spectra on their own, you are going to generate random variables out of those spectra (after proper normalization to turn the spectrum into a probability density function), and it is this bunch of random variables that you are going to cluster instead.
# %% [markdown]
# **II.1)** generate some scalar random variables from a given normalized spectrum, 

# %%
def secondDerivative(x):
    dev = []
    for i in range(len(x)):
        val = - 2 * x[i]
        if (i != len(x)-1):
            val+=x[i+1]
        if (i != 0):
            val+=x[i-1]
        dev.append(val)
    return dev


# %%
np.random.seed(50)


# %%
example = spectrums['LearningDataSet/1/spectre/1-1-007.pj2']
plt.figure(figsize=(15, 5))
plt.plot(example[0], example[1], label='density probability function', c='b')


# %%
from scipy.signal import argrelmax, find_peaks, peak_widths

peaks, _ = find_peaks(example[1])
results_half = peak_widths(example[1], peaks, rel_height=0.3)
h = results_half[1]
l = results_half[2] * (example[0].max() / (len(example[0]) - 1))
r = results_half[3] * (example[0].max() / (len(example[0]) - 1))
indices = []
for left, right in zip(l, r):
    indices_range = np.where((example[0] >= left) & (example[0] <= right))[0]
    indices.append(indices_range)
indices_source = np.concatenate(indices)
start = np.arange(0, indices_source.min())
indices_source = np.concatenate((start, indices_source))
source = example[0][indices_source]
proba = example[1][indices_source]
X_sample = np.sort(np.random.choice(source, 100, p=proba / np.sum(proba)))
Y_sample = [example[1][np.where(example[0]==val)[0]] for val in X_sample]
plt.figure(figsize=(15, 5))
plt.plot(example[0], example[1], label='density probability function', c='b')
plt.hlines(h, l, r, color="C2")
plt.scatter(np.squeeze(X_sample), Y_sample, c='r', label='random variables')


# %%
print(spectrums['LearningDataSet/12/spectre/12-2-005.pj2'])


# %%
example = spectrums['LearningDataSet/12/spectre/12-2-005.pj2']

def random_var(spectrum, samples=100):
    # threshold to only keep values on peaks
    X_sup = example[:, (np.where(spectrum[1]> 1.5)[0])]
    X = np.sort(np.random.choice(X_sup[0], samples, p=X_sup[1] / np.sum(X_sup[1])))
    Y = [X_sup[1][np.where(X_sup[0]==val)[0]] for val in X]
    return X, Y

print(example[0])
X, Y = random_var(example)


# %%
def get_random_var(X, Y):
    peaks, _ = find_peaks(Y)
    results_half = peak_widths(Y, peaks, rel_height=0.7)
    h = results_half[1]
    l = results_half[2] * (X.max() / (len(X) - 1))
    r = results_half[3] * (X.max() / (len(X) - 1))
    indices = []
    for left, right in zip(l, r):
        indices_range = np.where((X >= left) & (X <= right))[0]
        indices.append(indices_range)
    indices_source = np.concatenate(indices)
    start = np.arange(0, indices_source.min())
    indices_source = np.concatenate((start, indices_source))
    source = X[indices_source]
    proba = Y[indices_source]
    X_sample = np.sort(np.random.choice(source, 100, p=proba / np.sum(proba)))
    Y_sample = [Y[np.where(X==val)[0]] for val in X_sample]
    Y_sample = np.asarray([float(x) for x in Y_sample])
    return X_sample, Y_sample


# %%
plt.figure(figsize=(15, 5))
plt.plot(example[0], example[1], label='density probability function', c='b')
plt.scatter(np.squeeze(X), Y, c='r', label='random variables')
plt.xlabel("time (ms)")
plt.ylabel("intensity")
plt.legend()
plt.show()

# %% [markdown]
# **II.2)** for each NMR observation run a clustering algorithm to generate up to five clusters. 

# %%
X = np.expand_dims(X, axis=1)
X_Y = [[a[0], b[0]] for a, b in zip(X, Y)]
X_Y = np.array(X_Y)
X_Y = X_Y.reshape(-1, 2)
X_Y.shape


# %%
def get_pic_num(mod=None, repl=None, rank=None,key=None):
    if key:
        return np.sum(nmr_train.loc[nmr_train.loc[ : , 'sample' ] == key.split('/')[-1][:-4]].count()) - 1
    else:
        return np.sum(nmr_train.loc[nmr_train.loc[ : , 'sample' ] == f"{mod}-{repl}-"+rank].count()) - 1

# %% [markdown]
# ## Kmeans + inertia
# Here we tried running the kmeans with different cluster numbers and then picking the right number using the inertia with the elbow method or the silhouette score. This method ended up giving better results than the first one. 

# %%
#X = np.expand_dims(X, axis=1)
# X = np.squeeze(np.stack((X, np.array(Y)*X)))


# %%
pred=[]
inertia=[]
centers=[]
distortions=[]
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=1).fit(X)
    kmeans.fit(X) 
    centers.append(kmeans.cluster_centers_)
    pred.append(kmeans.predict(X))
    inertia.append(kmeans.inertia_)
    dist = cdist(X, kmeans.cluster_centers_, 'euclidean')
    distortions.append(sum(np.min(dist, axis=1)) / X.shape[0])

dev = secondDerivative(inertia)


# %%
i=3

plt.figure(figsize=(15, 5))
plt.plot(example[0], example[1])
plt.scatter(np.squeeze(X), Y, c=pred[i])


# %%
plt.figure(figsize=(15, 5))

# Elbow Method

dev = secondDerivative(inertia)
print(max(dev))
k_opt_e = np.where(dev==max(dev))[0][0]+2

plt.title('Elbow Method for optimal cluster number', size=20)
plt.plot(range(1, 10), inertia/max(inertia), 'bx-', markersize=10, label='inertia')
plt.axvline(x=k_opt_e, color='k', linestyle='--', label=f'{k_opt_e} clusters')

plt.plot(range(1, 10), distortions/max(distortions), 'rx-', markersize=10, label='distortion')
plt.ylabel('Kmeans inertia/distortion')
plt.xlabel('number of clusters')
plt.axvline(x=k_opt_e, color='k', linestyle='--', label=f'{k_opt_e} clusters')

plt.legend()
plt.xlim((0.8, 10))


plt.show()


# %%
plt.figure(figsize=(15, 5))

# Silhouette Method

sils = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X.reshape(-1, 1))
    label = kmeans.labels_
    sil_coeff = silhouette_score(X.reshape(-1, 1), label, metric='euclidean')
    sils.append(sil_coeff)
k_opt_s = np.argmax(sils) + 3
    
plt.plot(range(2, 10), sils, 'bx-', markersize=10)
plt.axvline(x=k_opt_s, color='k', linestyle='--', label=f'{k_opt_s} clusters')
plt.ylabel('silhouette score')
plt.xlabel('number of clusters')
plt.legend()
plt.xlim((0.8, 10)) 
plt.title('Silhouette coefficient method  for optimal cluster number', size=20)
plt.show()

# %% [markdown]
# **II.3)** Estimate the number of peaks. Create an evaluation rule to decide what is the best number of clusters to use (considering that you know the number of peaks thanks to the CSV) in order to minimize the sum of absolute difference between the estimated number of peaks and the ground truth number of peaks in an NMR observation.

# %%
nmr_train = pd.read_csv("LearningDataSet/data_nmr_input_epita_train.csv")
# nmr_train.loc[nmr_train.loc[ : , 'sample' ].str.match('10-1-*') == True]
nmr_train


# %%
def get_right_gt(spectrums, nmr_train):
    keys = []
    for key in spectrums.keys():
        keys.append(key[-12:-4])
    keys = [s.strip('/') for s in keys]
    mask = (nmr_train['sample'].isin(keys))
    nmr_train = nmr_train[mask]
    return nmr_train

nmr_train = get_right_gt(spectrums, nmr_train)


# %%
def get_right_keys(df):
    paths = glob.glob('LearningDataSet/*/spectre/*')
    keys = [x[-12:-4].strip('/') for x in paths]
    df = df[df.iloc[:, 0].isin(keys)]
    return df


# %%
def get_pic_num(nmr_train, row):
    return nmr_train.count(axis=1, numeric_only=True).iloc[row]


# %%
def get_label(nmr_train, row=None):
    if row == None:
        z = nmr_train.iloc[:, 1:].to_numpy()
        indices = np.where(~np.isnan(z))[1] + 1
        return indices
    else:
        z = pd.to_numeric(nmr_train.iloc[row, 1:]).to_numpy()
        indices = np.where(~np.isnan(z))[0] + 1
        return indices


# %%
def count_peaks(X, Y):
    X, Y = get_random_var(X, Y)
    X = np.expand_dims(X, axis=1)
    sils = []
    #silhouette
    for i in range(2, 10):
        kmeans = KMeans(n_clusters=i, random_state=1).fit(X.reshape(-1, 1))
        label = kmeans.labels_
        sil_coeff = silhouette_score(X.reshape(-1, 1), label, metric='euclidean', random_state=15)
        sils.append(sil_coeff)
    return min(np.argmax(sils) + 3, 5), X, Y


# %%
tot_err = 0
for l, key in enumerate(spectrums):
    example = spectrums[key]
    X, Y = get_random_var(example[0], example[1])
    X = np.expand_dims(X, axis=1)
    error_s = 0
    sils = []
    #silhouette
    for i in range(2, 6):
        error_s = 0
        kmeans = KMeans(n_clusters=i, random_state=1).fit(X.reshape(-1, 1))
        label = kmeans.labels_
        sil_coeff = silhouette_score(X.reshape(-1, 1), label, metric='euclidean', random_state=15)
        sils.append(sil_coeff)
    sil = np.argmax(sils) + 3
    real = get_pic_num(nmr_train, l)
    error_s += abs(real - sil)
    tot_err += error_s

print(f'TOTAL ERROR IS {tot_err} for {len(spectrums.keys())} peaks!')

# %% [markdown]
# ## Create a rule to merge the clusters
# Here, what we tried to do is to create ~40 clusters using the kmeans algorithm in order to make the distinction between the 1st and 2nd peaks that are often merged if you directly try to create less than 5 clusters. 
# Once we have these clusters, we need to create rules that enable us to merge the clusters we found. We try here to establish thresholds for the x and y axis that give proper results. 

# %%
kmeans = KMeans(n_clusters=40, random_state=1).fit(X_Y) 
kmcenters = kmeans.cluster_centers_
pred = kmeans.predict(X_Y)


# %%
pred_weight = [0] * 40
for j in range(40) :
    for i in range(len(pred)):
        if (pred[i] == j):
            pred_weight[j] += 1
            
def merge_clusters(cluster_centers, pred_weight, x_thresh=0.025, y_thresh=5):
    for i in range(len(cluster_centers)):
         for j in range(len(cluster_centers)):
            if j == i :
                continue
            wi = pred_weight[i]
            wj = pred_weight[j]
            if (wi == 0 or wj == 0):
                continue
            xi, xj = cluster_centers[i][0], cluster_centers[j][0]
            yi, yj = cluster_centers[i][1], cluster_centers[j][1]
            dist_x = abs(xi - xj)
            dist_y = abs(yi - yj)
            
            if dist_x < x_thresh or (dist_x < x_thresh + 1 and xi > 0.3 and xj > 0.3) :
                 if dist_y < y_thresh or (xi < 0.025 and xj < 0.025) or (dist_y < y_thresh + 1 and xi > 0.3 and xj > 0.3):
                    cluster_centers[i][0] = (cluster_centers[i][0] * wi + cluster_centers[j][0] * wj) / (wi + wj)
                    cluster_centers[i][1] = (cluster_centers[i][1] * wi + cluster_centers[j][1] * wj) / (wi + wj)
                    pred_weight[i] += wj
                    pred_weight[j] = 0
    centers = []
    for i in range(len(pred_weight)):
        if (pred_weight[i]):
            centers.append(cluster_centers[i])
    return centers

centers = merge_clusters(kmcenters, pred_weight)
len(centers),centers 


# %%
predictions = []
for i in range(len(X_Y)):
    min_err = abs(X_Y[i][0] - centers[0][0])
    min_ind = 0
    for idx, c in enumerate(centers):
        err = abs(c[0] - X_Y[i][0])
        if err < min_err : 
            min_err = err
            min_ind = idx
    predictions.append(min_ind)

plt.figure(figsize=(15, 5))
plt.plot(example[0], example[1])
plt.scatter(np.squeeze(X), Y, c=predictions)


# %%
tot_err = 0
for l, key in enumerate(spectrums):
    example = spectrums[key]
    X, Y = random_var(example)
    X = np.expand_dims(X, axis=1)
    X_Y = [[a[0], b[0]] for a, b in zip(X, Y)]
    X_Y = np.array(X_Y)
    X_Y = X_Y.reshape(-1, 2)
    error_s = 0
    kmeans = KMeans(n_clusters=40, random_state=1).fit(X_Y)
    centers = kmeans.cluster_centers_
    pred = kmeans.predict(X_Y)
    pred_weight = [0] * 40
    for j in range(40) :
        for i in range(len(pred)):
            if (pred[i] == j):
                pred_weight[j] += 1
    centers = merge_clusters(centers, pred_weight)
    real = get_pic_num(nmr_train, l)
    error_s += abs(real - max (3, min(len(centers), 5)))
    tot_err += error_s

print(f'TOTAL ERROR IS {tot_err} for {len(spectrums.keys())} peaks!')

# %% [markdown]
# ## Part 3: Classification

# %%
from sklearn.svm import SVC

# %% [markdown]
# Let's assume for a moment that you know the number of peaks in each observation. (you can count them from the csv file). The goal of this section is to classify these peaks from type 1 to type 5. 
# %% [markdown]
# **III.1)** generate some features from the clusters you created in Part 2 to classify peaks in their type (1, 2, 3, 4, 5).

# %%
def get_features(nb_peaks, X, Y):
    data = np.empty((1, 10))
    kmeans = KMeans(n_clusters=nb_peaks, random_state=1).fit(X)
    sample_silhouette_values = silhouette_samples(X, kmeans.labels_)
    a = kmeans.transform(X)
    a = np.take_along_axis(a, kmeans.labels_.reshape(-1, 1), axis=1)
    indexes = np.sort(np.unique(kmeans.labels_, return_index=True)[1])
    for unique, center in zip(kmeans.labels_[indexes], np.sort(kmeans.cluster_centers_, axis=0).ravel()):
        group = a[kmeans.labels_ == unique]
        group_y = Y[kmeans.labels_ == unique]
        group_x = X[kmeans.labels_ == unique]
        sil = sample_silhouette_values[kmeans.labels_ == unique].mean()
        sample = np.array([group.std(), group.shape[0], center, group_x.mean(), sil, group.sum(), group_x.std(), group.mean(), group_y.mean(), group_y.std()]).reshape(1, 10)
        data = np.concatenate((data, sample), axis=0)
    data = data[1:, :]
    centers = np.sort(kmeans.cluster_centers_.ravel()).reshape(1, -1)
    return data, centers


# %%
from sklearn.metrics import silhouette_samples, silhouette_score

pred = []
centers = []
distortions = []
distances = []
inertia = []
labels = []
data = np.empty((1, 10))
for l, key in enumerate(spectrums):
    # get spectrum
    example = spectrums[key]
    # generate random variables
    X, Y = get_random_var(example[0], example[1])
    Y = np.expand_dims(Y, axis=1)
    X = np.expand_dims(X, axis=1)
    # get number of peaks from csv
    peaks = get_pic_num(nmr_train, l)
    # cluster them
    kmeans = KMeans(n_clusters=peaks, random_state=1).fit(X)
    sample_silhouette_values = silhouette_samples(X, kmeans.labels_)
    a = kmeans.transform(X)
    a = np.take_along_axis(a, kmeans.labels_.reshape(-1, 1), axis=1)
    indexes = np.sort(np.unique(kmeans.labels_, return_index=True)[1])
    for unique, center in zip(kmeans.labels_[indexes], np.sort(kmeans.cluster_centers_.ravel())):
        group = a[kmeans.labels_ == unique]
        group_y = Y[kmeans.labels_ == unique]
        group_x = X[kmeans.labels_ == unique]
        sil = sample_silhouette_values[kmeans.labels_ == unique].mean()
        sample = np.array([group.std(), group.shape[0], center, group_x.mean(), sil, group.sum(), group_x.std(), group.mean(), group_y.mean(), group_y.std()]).reshape(1, 10)
        data = np.concatenate((data, sample), axis=0)
    kmeans = kmeans.fit(X)
    centers.append(np.sort(kmeans.cluster_centers_.ravel()))
    pred.append(kmeans.predict(X))
    inertia.append(kmeans.inertia_)
    dist = cdist(X, kmeans.cluster_centers_, 'euclidean')
    distances.append(dist)
    distortions.append(sum(np.min(dist, axis=1)) / X.shape[0])
    labels.append(kmeans.labels_)
centers = np.concatenate(centers).ravel()
data = data[1:, :]


# %%
print(data.shape)
print(centers.shape)


# %%
Y = get_label(nmr_train)
print(Y.shape)

# %% [markdown]
# **III.2)** What classification method would you use?
# %% [markdown]
# **III.3)** Run a classification method on each selected clusters and classify each peak you have identified. What is the accuracy of your method. Can you plot the confusion matrix ? 

# %%
from sklearn.model_selection import train_test_split

all_indices = list(range(len(Y)))
# prepare data for classification

## RAJOUTER COUNT LABELS, DISTANCES ET AUTRE INCH ##
#Y = np.expand_dims(np.concatenate(Y).ravel(), 1)

# centers
#centers = np.concatenate(centers).ravel()

# positions
#pos = np.concatenate(pos).ravel()

# number of labels per class
#temp = np.array([list(filter(lambda num: num != 0, lc[i])) for i in range(len(lc))])

#X = np.stack((centers, np.concatenate(temp).ravel()), axis=1)

# X = np.stack((centers, pos), axis=1)
X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.2, random_state=42)

#train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
#X_train = data[train_indices, :]
#X_test = data[test_indices, :]
#y_train = Y[train_indices]
#y_test = Y[test_indices]
#centers_test = centers[test_indices]


# %%
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

svm_model_linear = SVC(kernel='linear', random_state=0).fit(X_train, y_train)
features_names = ['group_std', 'group_shape[0]', 'center', 'group_x.mean()', 'sil', 'group.sum()', 'group_x.std()', 'group.mean()', 'group_y.mean()', 'group_y.std()']
pd.Series(abs(svm_model_linear.coef_[0]), index=features_names).nlargest(10).plot(kind='barh')
#svm_model_linear = MLPClassifier(hidden_layer_sizes=(400, 5), random_state=0).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test) 
  
# model accuracy for X_test
accuracy = svm_model_linear.score(X_test, y_test) 
  
# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions) 


# %%
print('accuracy : ', accuracy)


# %%
classes = [f'peak {i}' for i in range(1, 6)]
pd.DataFrame({classes[i]:cm[:,i] for i in range(len(classes))}, index=classes)


# %%
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, svm_predictions))

# %% [markdown]
# **III.4)** Create an estimator of the position of a peak. 

# %%
# input: spectre et peak number
from sklearn.metrics import mean_squared_error

spectrums = read_spectrum()
nmr_train = get_right_gt(spectrums, nmr_train)

centers = []
for l, key in enumerate(spectrums):
    # get spectrum
    example = spectrums[key]
    # generate random variables
    X, Y = get_random_var(example[0], example[1])
    #Y = np.asarray(Y)
    X = np.expand_dims(X, axis=1)
    # get number of peaks from csv
    peaks = get_pic_num(nmr_train, l)
    # cluster them
    kmeans = KMeans(n_clusters=peaks, random_state=1).fit(X)
    centers.append(list(np.sort(kmeans.cluster_centers_, axis=0).ravel()))

centers = np.concatenate(centers).ravel()

gt_peak_position = nmr_train.iloc[:, 1:].to_numpy().ravel()
gt_peak_position = gt_peak_position[~np.isnan(gt_peak_position)]
print(mean_squared_error(gt_peak_position, centers))

# %% [markdown]
# ## Part 4 : Putting things together. 
# %% [markdown]
# Part 2 was about counting the number of peaks while Part 3 was about characterizing peaks types. The goal of this section is to generate a complete workflow that estimates the peak types and positions existing in a given NMR spectrum. We suggest that you use Parts 2 and 3 to do so, but this is in no way mandatory.
# %% [markdown]
# In order to score this section we will run your function on a dataset that you don't have access to. But you can train your estimator on the training dataset you have access to (the data we use for testing are quite similar).
# 
# Your function should take a list of modalities, scrap all the existing replicates in these modalities folders and return a csv in the same shape as the one in the training dataset. 
# 
# Four loss functions will be used to evaluate the quality of your estimator:
# - The absolute difference between the number of detected peaks and the real number of peaks. 
# - For each detected peaks the accuracy of its classification type. 
# - For each detected peaks the square of the difference between the ground truth position and the estimated position. 
# 
# Your function should read a skeleton CSV of 6 columns: 
# The first column will indicate the modality-replicate-rank to read for the prediction. Then your function should write in the CSV your estimation of peak 1 to 5 position. 
# To help you, we created in advance the skeleton of the CSV. Then you should write your result in a new csv name "result.csv" stored in /app/Data/NMRBlue/LearningDataSet. 
# 
# It will be compared with the ground truth CSV with the following code:

# %%
def get_string(x):
    return np.asarray(x.split('-'))

def process(csv, classifier):
    df = pd.read_csv(csv)
    df = get_right_keys(df)
    first_column = df.iloc[:, 0].apply(get_string).values
    data = np.concatenate(first_column, axis=0).reshape(-1, 3)
    for row in range(data.shape[0]):
        X, Y = read_spectrum(*data[row, :])
        X, Y, scale = normalize(X, Y)
        nb_peaks, X_sample, Y_sample = count_peaks(X, Y)
        feature, centers = get_features(nb_peaks, X_sample, Y_sample)
        centers *= scale
        peak_numbers = classifier.predict(feature).reshape(1, -1)
        indices = np.unique(peak_numbers, return_index=True)[1]
        peak_numbers = peak_numbers[:, indices]
        centers = centers[:, indices]
        a = np.empty((1, 5))
        a[:] = np.nan
        a[:, peak_numbers - 1] = centers
        df.iloc[row, 1:] = np.squeeze(a, axis=0)
    df.to_csv('result.csv')
    return df

df = process('LearningDataSet/data_nmr_input_epita_train.csv', svm_model_linear)
df


# %%
import pandas as pd
import sklearn
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


pred_peak = pd.read_csv("/app/Data/NMRBlue/LearningDataSet/result.csv", index_col=0)
gt_peak = pd.read_csv("/app/Data/NMRBlue/LearningDataSet/gt.csv", index_col=0)


# %%
def counting_number_peak(df):
    res = df.count(axis=1, level=None, numeric_only=True)
    return res

def mae_count(gt, pred):
    count_gt = counting_number_peak(gt)
    count_pred = counting_number_peak(pred)
    mae = mean_absolute_error(count_gt, count_pred)
    return mae


# %%
# Score MAE
mae_count(gt_peak, pred_peak)


# %%
def flat(df):
    flatten = df.to_numpy().flatten()
    return flatten
def flat_boolean(df):
    flatten = flat(df)
    bool_flatten = ~np.isnan(flatten)
    return bool_flatten
    


# %%
boolean_gt = flat_boolean(gt_peak)
boolean_pred = flat_boolean(pred_peak)
accuracy_score(boolean_gt, boolean_pred)


# %%
flat_gt = flat(gt_peak)
flat_pred = flat(pred_peak)
not_nan_gt = ~np.isnan(flat_gt)
flat_gt = flat_gt[not_nan_gt]
flat_pred = flat_pred[not_nan_gt]

not_nan_pred = ~np.isnan(flat_pred)
flat_gt = flat_gt[not_nan_pred]
flat_pred = flat_pred[not_nan_pred]

sklearn.metrics.mean_squared_error(flat_gt, flat_pred)

# %% [markdown]
# ## Optional part: Bayesian or Deep Learning peak detection
# For those who want to, you could apply this if you want to dig bayesian stat:
# https://www.researchgate.net/publication/258252347_Bayesian_Peak_Picking_for_NMR_Spectra
# Or this if you want to dig CNN in NMR (adjustment to single dimension will be needed)
# https://academic.oup.com/bioinformatics/article/34/15/2590/4934937
# 

