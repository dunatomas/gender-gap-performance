# Women Do Better in Sports

<p align="center">
  <img src="https://github.com/user-attachments/assets/46197b2f-1470-49de-a491-6172895ab256" width="800" alt="Final Project Preview"/>
</p>

## Exploring the narrowing performance gap between men and women

This repository contains the code and data pipeline developed for my Master’s Thesis, which analyzes how the performance gap between women and men has evolved over time across comparable athletics and swimming disciplines.

Rather than focusing on absolute performance differences, the project emphasizes **rates of improvement** and **historical progression**, showing that women have often improved faster than men once comparable competitive conditions were established. The results are communicated through an **interactive Streamlit dashboard** designed for both in-depth inspection and cross-discipline comparison.

🔗 **Live dashboard**:
[Women Do Better in Sports – Interactive visualization](https://women-do-better-in-sports.streamlit.app/)


## 🔍 Core idea

For each discipline (e.g., 100 m sprint, marathon, swimming freestyle):

* Build **historical best-so-far progressions** of women’s and men’s world records.
* Compare improvement dynamics over time, highlighting differences in **progression speed** rather than static gaps.
* Introduce a historically grounded **men–women gap framing**, showing how far back in the men’s record timeline the current women’s record would rank.
* Extend record trajectories using a **normalized saturation-based prediction model**, producing plausible near-limit trends rather than linear extrapolations.


## 📊 Interactive visualizations

The Streamlit app provides two complementary views:

### 1️⃣ Single-discipline view

- Detailed inspection of one event at a time
- Historical record progression for women and men
- Optional overlays:
  - Gap / crossing reference line
  - Regression slope indicators
  - Near-limit predictive trajectories
 
<p align="center">
  <img src="https://github.com/user-attachments/assets/209b8184-3ac3-4045-a67d-fb1c1c6b7d54" width="800" alt="Single-discipline View Example"/>
</p>

### 2️⃣ Multi-discipline grid view

* Mini-plots for all disciplines shown simultaneously
* Filters by **category** (running, swimming, jumps) and **subcategory**
* Sorting by **women’s improvement advantage relative to men** (percentage-based)
* Filters to identify disciplines where women have or have not reached comparable historical men’s levels

This grid view makes it possible to detect **systematic patterns**, such as the strong concentration of women’s faster improvement in endurance running disciplines.

<p align="center">
  <img src="https://github.com/user-attachments/assets/618d41e5-8256-4b88-a904-f22808e616b0" width="800" alt="Grid View Example"/>
</p>

## 🧠 Modeling philosophy

Predictive components are included for **exploratory and visual support**, not for precise forecasting. Instead of event-specific extrapolations, the project uses a **normalized universal saturation model** that:

* Enforces bounded long-term improvement
* Stabilizes predictions across disciplines with sparse or truncated histories
* Aligns with established evidence of physiological and technological limits in elite sport

Predictions illustrate how the gender gap may continue to **narrow gradually**, though at much lower rates than those observed during the rapid expansion of women’s sport in the 20th century.


## 🛠️ Repository structure

* `data/`
  * `raw/` – original record progression sources
  * `processed/` – cleaned and structured datasets
  * `predictions` - sports predictions
* `notebooks/` – data cleaning, exploration, modeling and prediction
* `app.py` – Streamlit application


## 🎓 Context

This project builds upon the *gender-o-meter* concept by Jaume Nualart and Mar Canet, extending it into a fully reproducible analysis pipeline with predictive modeling and large-scale comparative visualization.

It was developed as part of a Master’s Thesis in data science and visualization, with an explicit focus on **gender equity**, **historical context**, and **responsible interpretation of performance data**.


## 📄 License and data

All data sources used are publicly available.
The code is provided for academic and educational purposes.

