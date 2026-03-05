# Machine_learning_Lab_7_PCA-MDS-K-means-Security-Analytics
Unsupervised ML from scratch: PCA (SVD/Eigen), MDS, k-means &amp; soft k-means with visualizations—security analytics/anomaly detection foundations.
# Unsupervised ML for Security Analytics — PCA, MDS, K-Means (From Scratch)

This project implements core **unsupervised learning** techniques commonly used in cybersecurity analytics:
- **PCA** (Principal Component Analysis) via both **SVD** and **Eigen-decomposition**
- **MDS** (Multidimensional Scaling) for distance-based embeddings
- **K-Means** and **Soft K-Means** clustering for grouping unlabeled data

These techniques help reduce noise, visualize complex telemetry, and identify patterns such as anomalies or suspicious clusters.

---

## Why this matters for Cybersecurity + Cloud Security
In cloud environments, security teams often analyze high-volume, high-dimensional telemetry:
- VPC Flow Logs / network traffic statistics
- CloudTrail event features
- endpoint behavior signals
- authentication and identity patterns

Unsupervised methods help when you **don’t have labels** (common in real incidents):
- cluster similar events/users/hosts
- detect outliers (possible intrusions or fraud)
- reduce dimensionality for dashboards and triage

---

## What’s Implemented

### 1) PCA (Principal Component Analysis)
- Standardization using `StandardScaler`
- Covariance computation:
- vanilla loop implementation
- fast matrix multiplication: Σ = (1/m) XᵀX
- Principal components via:
- **SVD** on (1/√m)X
- **Eigen** decomposition on Σ
- Visualization:
- Iris dataset projected onto top 2 PCs
- Reconstruction:
- Optdigits dataset reconstruction error vs number of PCs (k=1..64)

**Security analogy:** PCA compresses telemetry features into fewer dimensions while preserving variance — useful for building lightweight anomaly detectors or visual triage views.

---

### 2) MDS (Multidimensional Scaling)
- Implemented classical MDS using distance matrix → embedding (k=2)
- Applied to European city distance matrix to produce a 2D map

**Security analogy:** MDS is useful when you only have distances/similarities (e.g., behavioral similarity scores, graph distances between entities, or embedding distances).

---

### 3) Clustering (K-Means + Soft K-Means)
- K-Means from scratch:
- center initialization from random examples
- iterative assignment using `cdist`
- update centers by cluster means
- stopping via tolerance on center movement
- Membership prediction (hard labels)
- Soft K-Means from scratch:
- soft responsibilities using exp(-||x - m|| / beta)
- beta set using median pairwise distances (scaled)
- Soft membership prediction (probabilities)
- Visualization:
- blobs dataset with k in {2,3,5}
- failure cases (two moons, stretched Gaussians)

**Security analogy:** clustering can group similar behaviors (e.g., normal vs suspicious login clusters). The failure cases illustrate why choosing k and geometry matters — important for unsupervised threat detection.

---

## Datasets Used
- Iris (UCI)
- Optdigits (UCI)
- Synthetic datasets:
- blobs
- two moons
- stretched Gaussian

