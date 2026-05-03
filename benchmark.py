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

def generate_llm_executive_summary(results):
    """
    GEN AI INTEGRATION HOOK:
    Formats the benchmark metrics into a prompt intended for an LLM (e.g., Llama 3, Gemini, OpenAI).
    In a full production environment, this text is sent via API to generate an autonomous MLOps report.
    """
    print("\n" + "="*50)
    print("🤖 INITIATING GEN AI AUTONOMOUS EVALUATION AGENT")
    print("="*50)
    
    rf_acc = results['rf_acc'][-1] * 100
    gbt_acc = results['gbt_acc'][-1] * 100
    rf_time = results['rf_time'][-1]
    gbt_time = results['gbt_time'][-1]
    
    prompt = f"""
    [SYSTEM INSTRUCTION]
    You are a Lead MLOps Engineer Agent. Analyze the following Apache Spark distributed training logs.
    Compare Bagging vs. Boosting focusing on distributed network I/O and synchronization penalties.
    
    [SPARK EXECUTION LOGS - 100% DATA SCALE]
    - Algorithm A (Random Forest): Accuracy = {rf_acc:.2f}%, Training Time = {rf_time:.2f} seconds.
    - Algorithm B (Gradient Boosted Trees): Accuracy = {gbt_acc:.2f}%, Training Time = {gbt_time:.2f} seconds.
    
    [TASK]
    Generate a 3-sentence executive summary detailing the ROI of compute time versus accuracy gains.
    """
    
    print("\n[Generated LLM Prompt Payload]")
    print(prompt)
    print("\n[Mock LLM Response]")
    print(f"Executive Summary: Gradient Boosted Trees achieved a higher accuracy of {gbt_acc:.2f}%, outperforming Random Forest by capturing complex non-linear particle interactions. However, GBT suffered a significant parallelism penalty, requiring {gbt_time - rf_time:.2f} additional seconds of compute time due to sequential network synchronization. Organizations must weigh this ~18% increase in cloud infrastructure cost against the strict necessity of the 3% accuracy gain.")
    print("="*50 + "\n")

def run_benchmark(algorithm, dataset, train_ratio, target_col):
    sampled_data = dataset.sample(withReplacement=False, fraction=train_ratio, seed=42)
    train, test = sampled_data.randomSplit([0.8, 0.2], seed=42)

    start_time = time.time()
    model = algorithm.fit(train)
    end_time = time.time()

    predictions = model.transform(test)
    evaluator = MulticlassClassificationEvaluator(labelCol=target_col, metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    return end_time - start_time, accuracy

def main():
    print("Initializing Distributed Benchmark Pipeline...")
    ensure_dir("figures")

    spark = SparkSession.builder \
        .appName("RF_vs_GBT_Scalability") \
        .config("spark.driver.memory", "10g") \
        .config("spark.executor.memory", "10g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    file_path = "HIGGS.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please download and unzip the HIGGS dataset.")
        spark.stop()
        return

    print("Loading Data into Spark RDDs...")
    df = spark.read.csv(file_path, header=False, inferSchema=True)
    
    old_columns = df.columns
    target_col = "label"
    df = df.withColumnRenamed(old_columns[0], target_col)

    # Preprocessing
    print("Executing Vector Assembler...")
    feature_cols = [c for c in df.columns if c != target_col]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_clean = df.na.drop()
    data = assembler.transform(df_clean).select("features", target_col)

    # Experiment Loop
    scales = [0.2, 0.5, 1.0]
    results = {'scale': [], 'rf_time': [], 'gbt_time': [], 'rf_acc': [], 'gbt_acc': []}

    for scale in scales:
        print(f"\n--- Processing Data Scale: {scale * 100}% ---")

        print("Training Random Forest...")
        rf = RandomForestClassifier(labelCol=target_col, featuresCol="features", numTrees=20, maxDepth=5)
        rf_time, rf_acc = run_benchmark(rf, data, scale, target_col)
        print(f"RF Finished in {rf_time:.2f}s | Acc: {rf_acc:.4f}")

        print("Training Gradient Boosted Trees...")
        gbt = GBTClassifier(labelCol=target_col, featuresCol="features", maxIter=20, maxDepth=5)
        gbt_time, gbt_acc = run_benchmark(gbt, data, scale, target_col)
        print(f"GBT Finished in {gbt_time:.2f}s | Acc: {gbt_acc:.4f}")

        results['scale'].append(scale * 100)
        results['rf_time'].append(rf_time)
        results['gbt_time'].append(gbt_time)
        results['rf_acc'].append(rf_acc)
        results['gbt_acc'].append(gbt_acc)

    # Gen AI Reporting Hook
    generate_llm_executive_summary(results)

    spark.stop()

if __name__ == "__main__":
    main()
