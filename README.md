# 📈 Distributed Ensemble ML Benchmark: Bagging vs. Boosting in Apache Spark

A comprehensive scalability and performance analysis of Random Forest (RF) and Gradient Boosted Trees (GBT) implemented within the Apache Spark ecosystem, benchmarked against the 11-million instance HIGGS dataset.

### 🌟 Benchmark Overview

Data scientists operating in large-scale cluster environments face a constant trade-off between model performance and computational efficiency. This project quantifies the relationship between dataset volume and training latency in a distributed environment, highlighting the "Parallelism Penalty" inherent in sequential boosting algorithms.

* **⚡ Linear Scalability:** Demonstrates how Random Forest utilizes Bagging (Bootstrap Aggregating) to achieve near-linear scalability, making it ideal for massive, distributed datasets.
* **📉 The Cost of Synchronization:** Analyzes the super-linear time complexity and network I/O bottlenecks of Gradient Boosted Trees due to their sequential residual-error optimization.
* **📊 Big Data Benchmarking:** Utilizes PySpark to distribute and process the 2.6GB HIGGS classification dataset (11,000,000 instances, 28 features).
* **⚖️ ROI Analysis:** Provides a cost-benefit framework for cloud computing environments, measuring if the marginal accuracy gains of GBT justify the substantial increase in compute time and infrastructure fees.

## 🏗️ Experimental Design

The experiment is conducted using PySpark MLlib to handle distributed memory management (RDDs), scaling beyond single-machine RAM constraints. 

To prevent tuning bias, structural complexity was controlled:
* **Tree Depth:** Fixed at 5 for both models to ensure baseline construction times are equivalent.
* **Iterations/Trees:** Fixed at 20. RF builds 20 independent trees; GBT builds 20 sequential trees.
* **Data Scaling:** Models were evaluated at 20%, 50%, and 100% data loads to establish growth trends and full-load performance metrics.

### 🏆 Final Benchmark Results (100% Data Scale)

| Algorithm | Training Time | Accuracy | Scalability Profile |
| :--- | :--- | :--- | :--- |
| **Random Forest** | 1493.62 seconds | 67.35% | CPU-Bound (Linear) |
| **Gradient Boosting** | 1764.62 seconds | 70.52% | I/O-Bound (Super-Linear) |

*Gradient Boosting required ~18% more training time at full scale due to the network latency of re-computing and broadcasting error gradients across the cluster after every iteration.*

## 💻 Tech Stack

* **Language:** Python 3.12
* **Distributed Computing Framework:** Apache Spark (PySpark 4.0.2)
* **Machine Learning:** PySpark MLlib (RandomForestClassifier, GBTClassifier)
* **Data Manipulation & Visualization:** Pandas, Matplotlib

## 💻 Quickstart & Development

**1. Clone the repository:**
```bash
git clone [https://github.com/amjad-hanini/Spark-Ensemble-ML-Benchmark.git](https://github.com/amjad-hanini/Spark-Ensemble-ML-Benchmark.git)
cd Spark-Ensemble-ML-Benchmark
```

**2. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

**3. Download the HIGGS Dataset:**
Ensure `wget` is installed, then download the dataset to the project root:
```bash
wget -nc [https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz](https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz)
gunzip -k HIGGS.csv.gz
```

**4. Execute the Benchmark Pipeline:**
```bash
python benchmark.py
```
*Note: Full execution on the 100% data scale may take 2-3 hours depending on your local machine or cluster configuration.*

## 🚀 Roadmap & Future Scaling

* **Hyperparameter Grid Search:** Implement PySpark's `CrossValidator` to find optimal depth-to-iteration ratios without inducing extreme memory pressure.
* **Advanced Boosting Implementations:** Integrate `XGBoost4J-Spark` to evaluate if modern histogram-based approximation methods mitigate the network I/O bottlenecks observed in standard Spark GBT.
* **Cloud Deployment:** Package the benchmark into a `.jar` or Docker container for native execution on AWS EMR or Databricks clusters.

## 📝 License

MIT License - see LICENSE for details.

## 👨‍💻 Author

**Amjad Hanini**
* GitHub: [@amjad-hanini](https://github.com/amjad-hanini)
