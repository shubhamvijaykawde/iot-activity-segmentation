# Industrial IoT Activity Segmentation using ClaSP

Unsupervised segmentation pipeline for detecting machine activity regimes in industrial IoT time-series data.
The system transforms heterogeneous multi-sensor streams into a robust 1-D signal and detects behavioral state transitions using **ClaSP change-point detection**.

---

## Motivation

Industrial production lines generate continuous sensor streams but often lack reliable activity labels.
Understanding machine behavior requires identifying **when the process changes**, not just predicting values.

This project focuses on:

* Discovering machine activity boundaries
* Detecting process regime shifts
* Evaluating segmentation quality without reliable ground truth

---

## Key Features

* **Custom IoT data loader** for JSONL factory logs
* **Sensor type aware feature engineering** (binary, continuous, position sensors)
* **Signal aggregation strategy** to combine heterogeneous sensors into a stable representation
* **ClaSP ensemble segmentation** for unsupervised change-point detection
* **Baseline comparison (PELT – ruptures)**
* **Weak-label evaluation metrics** (since real labels are unreliable)
* **Quantitative segmentation quality scoring**

---

## Project Structure

```
iot-activity-segmentation/
│
├── src/
│   ├── IOTLoader.py
│   ├── FeatureEngineer.py
│   ├── Segmentation.py
│   ├── CBRExporter.py
│   └── RunPipeline.py
│
├── Complete_Final_Segmentation.ipynb
│   
│
├── data/
│   └── README.md
│
├── segmentation_output/
│   └── .gitkeep
│
├── requirements.txt
├── README.md
├── .gitignore

```

---

## Dataset

Download the dataset from:

https://zenodo.org/records/15773369

After downloading,  Navigate to:  
   `data-radiant-eval-paper/factory/evaluation/iot_logs/`

You will find **5 JSONL files** containing IoT sensor logs from industrial machinery.

In the code we use this location :- data/data-radiant-eval-paper/factory/evaluation/iot_logs


The dataset is not included due to size and licensing constraints.

---

## Installation

### 1. Clone repository

```
git clone https://github.com/shubhamvijaykawde/iot-activity-segmentation.git
cd iot-activity-segmentation
```

### 2. Create environment (Windows PowerShell)

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## Running the Pipeline

### Option 1 — Full pipeline

```
python src/RunPipeline.py
```

Outputs will be saved in:

```
segmentation_output/
```

### Option 2 — Interactive exploration

Run the notebook:

```
notebooks/Complete_Final_Segmentation.ipynb
```

The notebook allows:

* Step-by-step execution
* Visualization of signals
* Boundary inspection
* Metric analysis

---

## Method Overview

### 1. Signal Construction

Multi-sensor streams are converted into a single segmentation signal:

| Sensor Type | Processing                  |
| ----------- | --------------------------- |
| Binary      | rolling stabilization       |
| Continuous  | min-max normalization       |
| Position    | motion magnitude extraction |

Final signal = weighted aggregated representation

---

### 2. Segmentation

Primary algorithm:
**ClaSP (Classification Score Profile)**

Fallback:
**Recursive binary segmentation**

Baseline:
**PELT (ruptures)**

---

### 3. Evaluation Without Ground Truth

Because industrial logs contain noisy labels, segmentation quality is evaluated using structural metrics:

* Boundary strength score
* Segment stability score
* Separation score
* Explained variance
* Regime consistency

This evaluates segmentation *structure*, not just classification accuracy.

---

## Use Cases

* Process monitoring
* Anomaly detection preprocessing
* Predictive maintenance feature extraction
* Activity discovery in unlabeled IoT environments

---

## Technologies

Python, Pandas, NumPy, ClaSP, Ruptures, Scikit-Learn, Matplotlib

---

## License

MIT License
