import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_benchmark(algorithm, dataset, train_ratio, target_col):
    """
    Samples data, splits into train/test, and measures training time and accuracy.
    """
    sampled_data = dataset.sample(withReplacement=False, fraction=train_ratio, seed=42)
    train, test = sampled_data.randomSplit([0.8, 0.2], seed=42)

    # Measure Training Time
    start_time = time.time()
    model = algorithm.fit(train)
    end_time = time.time()

    # Measure Accuracy
    predictions = model.transform(test)
    evaluator = MulticlassClassificationEvaluator(labelCol=target_col, metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    return end_time - start_time, accuracy

def main():
    print("Initializing Distributed Benchmark Pipeline...")
    ensure_dir("figures")

    # Step 1: Initialize Spark Session
    spark = SparkSession.builder \
        .appName("RF_vs_GBT_Scalability") \
        .config("spark.driver.memory", "10g") \
        .config("spark.executor.memory", "10g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    # Step 2: Load Dataset
    file_path = "HIGGS.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please download and unzip the HIGGS dataset.")
        spark.stop()
        return

    print("Loading Data into Spark RDDs...")
    df = spark.read.csv(file_path, header=False, inferSchema=True)
    
    # The UCI HIGGS dataset has no header. First column is the target label.
    old_columns = df.columns
    target_col = "label"
    df = df.withColumnRenamed(old_columns[0], target_col)

    # Generate Figure 1: Target Variable Distribution
    print("Generating Figure 1: Class Distribution...")
    class_counts = df.groupBy(target_col).count().toPandas().sort_values(target_col)
    
    plt.figure(figsize=(8, 5))
    plt.bar(class_counts[target_col].astype(str), class_counts['count'], color=['#4c72b0', '#dd8452'])
    plt.title('Figure 1: Class Distribution of HIGGS Dataset')
    plt.xlabel('Class Label (0 = Background, 1 = Signal)')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('figures/Figure1.png')
    plt.close()

    # Step 3: Preprocessing
    print("Executing Vector Assembler...")
    feature_cols = [c for c in df.columns if c != target_col]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_clean = df.na.drop()
    data = assembler.transform(df_clean).select("features", target_col)

    # Step 4: Experiment Loop
    scales = [0.2, 0.5, 1.0]
    results = {'scale': [], 'rf_time': [], 'gbt_time': [], 'rf_acc': [], 'gbt_acc': []}

    for scale in scales:
        print(f"\n--- Processing Data Scale: {scale * 100}% ---")

        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(labelCol=target_col, featuresCol="features", numTrees=20, maxDepth=5)
        rf_time, rf_acc = run_benchmark(rf, data, scale, target_col)
        print(f"RF Finished in {rf_time:.2f}s | Acc: {rf_acc:.4f}")

        # Gradient Boosted Trees
        print("Training Gradient Boosted Trees...")
        gbt = GBTClassifier(labelCol=target_col, featuresCol="features", maxIter=20, maxDepth=5)
        gbt_time, gbt_acc = run_benchmark(gbt, data, scale, target_col)
        print(f"GBT Finished in {gbt_time:.2f}s | Acc: {gbt_acc:.4f}")

        results['scale'].append(scale * 100)
        results['rf_time'].append(rf_time)
        results['gbt_time'].append(gbt_time)
        results['rf_acc'].append(rf_acc)
        results['gbt_acc'].append(gbt_acc)

    # Generate Figure 2: Training Time
    print("\nGenerating Figure 2: Training Time Scalability...")
    plt.figure(figsize=(10, 6))
    plt.plot(results['scale'], results['rf_time'], marker='o', label='Random Forest', linewidth=2)
    plt.plot(results['scale'], results['gbt_time'], marker='s', label='Gradient Boosting', linewidth=2)
    plt.title('Figure 2: Training Time vs. Data Scale')
    plt.xlabel('Data Size (%)')
    plt.ylabel('Training Time (Seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/Figure2.png')
    plt.close()

    # Generate Figure 3: Accuracy
    print("Generating Figure 3: Final Model Accuracy...")
    final_rf_acc = results['rf_acc'][-1]
    final_gbt_acc = results['gbt_acc'][-1]

    plt.figure(figsize=(7, 5))
    plt.bar(['Random Forest', 'Gradient Boosting'], [final_rf_acc, final_gbt_acc], color=['blue', 'green'])
    plt.ylim(0.5, 0.8)
    plt.title('Figure 3: Final Model Accuracy Comparison (100% Scale)')
    plt.ylabel('Accuracy Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('figures/Figure3.png')
    plt.close()

    # Final Output
    print("\n===================================")
    print("   FINAL BENCHMARK RESULTS (100%)  ")
    print("===================================")
    print(f"RF Time:         {results['rf_time'][-1]:.2f} seconds")
    print(f"GBT Time:        {results['gbt_time'][-1]:.2f} seconds")
    print(f"RF Accuracy:     {final_rf_acc:.4f}")
    print(f"GBT Accuracy:    {final_gbt_acc:.4f}")
    print("===================================\n")

    spark.stop()

if __name__ == "__main__":
    main()
