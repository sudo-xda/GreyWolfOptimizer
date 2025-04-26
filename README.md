# 🐺 Grey Wolf Optimizer (GWO) – Python Port from MATLAB

> A minimalist and powerful **Grey Wolf Optimizer (GWO)** in Python, originally developed in MATLAB and now ported for easy use in Pythonic AI/ML workflows. Ideal for **filter optimization**, **hyperparameter tuning**, and other global optimization problems.

![gwo-banner](https://ch.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/44974/versions/9/screenshot.jpg)

---

## 📌 Key Features

✅ Simple and clean implementation  
✅ Inspired by social hierarchy and hunting behavior of grey wolves  
✅ Optimizes any custom loss/objective function  
✅ Easy to plug into machine learning pipelines  
✅ Only depends on **NumPy**  
✅ Fully customizable for research or production use

---

## 🌐 Algorithm Inspiration

Grey Wolf Optimizer (GWO) mimics the leadership structure and cooperative hunting of grey wolves:
- **Alpha (α)** – best solution
- **Beta (β)** – second-best
- **Delta (δ)** – third-best
- **Omega (ω)** – the rest

Each wolf updates its position based on α, β, and δ, encouraging **exploration** and **exploitation**.

📖 *Reference*:  
Mirjalili, S., Mirjalili, S.M., & Lewis, A. (2014). [Grey Wolf Optimizer](https://doi.org/10.1016/j.advengsoft.2013.12.007), *Advances in Engineering Software*.

---

## 🧠 How It Works

1. Randomly initialize a pack of grey wolves (candidate solutions)
2. Identify the top 3 solutions: alpha, beta, and delta
3. Update all other solutions using a weighted influence of the top 3
4. Repeat for a number of iterations to converge on the global minimum

---

## 🛠️ Installation

You only need **NumPy**:

```bash
pip install numpy
```
```python
from gwo import GWO
import numpy as np
```
# 🎯 Define objective function (Sphere function example)
```
def sphere(x):
    return np.sum(x ** 2)
```
# ⚙️ Create GWO instance
```
gwo = GWO(
    obj_function=sphere,      # Your custom loss/objective function
    lb=[-5]*10,               # Lower bounds per dimension
    ub=[5]*10,                # Upper bounds per dimension
    dim=10,                   # Dimensionality of problem
    n_agents=30,              # Number of grey wolves (population)
    max_iter=100              # Total number of iterations
)
```
# 🐺 Run the optimization
```
best_position, best_score = gwo.run()

print("✅ Best position:", best_position)
print("🎯 Best score:", best_score)
