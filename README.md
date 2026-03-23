# 🧠 One-Bit Unlimited Sampling & Modulo ADC

## 📌 Overview
This project explores **unlimited sampling using modulo ADCs**, a framework that overcomes the **finite dynamic range limitation** of traditional ADCs.

Conventional ADCs suffer from **clipping and saturation** when signals exceed range. This work demonstrates how **modulo folding + reconstruction algorithms** can recover high dynamic range signals accurately.

---

## 🚨 Problem Statement
In practical ADC systems:
- Signals exceeding range → **information loss**
- Increasing range → **hardware complexity**

👉 Goal:
- Encode signals into bounded range  
- Reconstruct original signal with minimal error  

---

## 💡 Key Idea: Modulo Sampling

Instead of clipping, we apply:
<p align="center">
  <img src="Screenshot 2026-03-23 233253.png" width="350" title="hover text">
</p>


- Folds signal into range **[-λ, λ]**
- Retains only **least significant information**
- Discards amplitude overflow (temporarily)


---

## 🔄 Trade-off

| Advantage | Drawback |
|----------|--------|
| Low hardware complexity | Ambiguity in samples |
| One-bit possible | Requires reconstruction |

👉 Ambiguity resolved using **oversampling**

---

## ⚙️ One-Bit Sampling

Extreme case:
- Only **1-bit (±1)** is stored
- Acts like a **comparator / sign function**

Yet, reconstruction is still possible!

---

## 📡 Signal Assumptions

- Signal is **bandlimited (Ω)**
- Oversampling is used
- Superoscillations may occur

---

## 📊 Visualizations

### 🔹 B2R2 OUTPUT
![B2R2 OUTPUT]("Screenshot 2026-03-23 233313.png")
## SYMBOL ERROR RATE MONTECARLO SIMULATIONS
![SER]("Screenshot 2026-03-23 232905.png")
👉 Unlike conventional sampling, modulo sampling **captures more structure** of high-amplitude signals.

---

## ⚡ Conventional vs Modulo Sampling

### 🔻 Direct Quantization (Fails)
- Saturation occurs
- Information loss

### 🔺 Modulo Quantization
- No saturation
- Encodes overflow implicitly

![Comparison](images/comparison.png)

---

## 🔁 Recovery Algorithm

### Step 1: Averaging (Low-pass filtering)
- Smooth quantized samples
- Implemented using **B-splines**
- Reduces noise and variation

👉 Analogy: Image smoothing filter

---

### Step 2: High-pass Filtering
- Detect sharp transitions (folding points)
- Equivalent to **edge detection**

---

### Step 3: Reconstruction
- Recover lost multiples of **2λ**
- Use signal structure + oversampling

---

## 🧮 Advanced Recovery (B2R2)

We solve:
Fλ = VZ
Where:
- V = Vandermonde matrix  
- Z = unknown signal  

---

### 🚀 Optimization Approach

We use **Projected Gradient Descent (PGD)**:

---

### 🎯 Intuition (Ball Rolling Analogy)

- Signal = ball  
- Loss function = terrain  
- Constraint = fence  

Ball rolls → projected back → converges

---



📌 Observation:
- Error drops drastically after certain OF

---


📌 Observation:
- Approaches **1 with higher oversampling**


📌 Observation:
- B2R2 significantly outperforms direct methods


## 🔬 Key Insights

- Modulo ADC avoids saturation completely  
- Oversampling resolves ambiguity  
- One-bit sampling still retains recoverable structure  
- Reconstruction behaves like **inverse problem + optimization**

---

## ⚠️ Limitations

- Requires oversampling  
- Sensitive to noise  
- Matrix inversion can be unstable  
- Some theoretical assumptions approximated  

---

## 🔗 Code & Implementation

👉 Colab Notebook: *(add your link here)*  

---

## 📚 References

- Unlimited Sampling Theory  
- Modulo ADC Papers  
- Signal Processing Fundamentals  

---

## ✍️ Author

**Krunal Vaghela**  
- Signal Processing | ML | Research  

---

## ⭐ If you like this work

Give a ⭐ to the repo!
