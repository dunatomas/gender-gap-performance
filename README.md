# Gender Gap in Sports Performance

**Do women do better?**  
This project explores how the performance gap between men and women has evolved across time in different sports disciplines, with a special focus on **world records** and moments where **women's records cross past men's records** ("glass ceiling" effect).

The project is part of my Master's Thesis and aims to produce:

- An **interactive Streamlit app** to explore records, predictions, and gender gaps.
- A **storytelling landing page** with scrollytelling visuals (for portfolio / competitions such as *Information is Beautiful*).

---

## 🔍 Core idea

For each discipline (e.g. 100m sprint):

- Plot the **historical progression of men's and women's world records** (time or distance).
- Highlight the **crossover moment**:  
  when the **current women's record** is **faster/better than all men's records before a certain year**.
- Extend the curves into the **future** using prediction models (ML/DL)  
  – shown as dashed lines or uncertainty bands to distinguish them from observed data.

This allows statements like:

> "Before **1930**, no man had ever run as fast as the **current women’s 100m world record**."

---

## 🧱 Project structure

```text
gender-gap-performance/
│
├── README.md                       # Project description (this file)
├── requirements.txt                # Python dependencies
│
├── app/                            # STREAMLIT APPLICATION
│   ├── app.py                      # Main Streamlit entry point
│   ├── pages/                      # (Optional) extra pages
│   ├── components/                 # (Optional) reusable plotting / UI components
│   ├── assets/                     # Logos, custom CSS, images
│   └── __init__.py
│
├── landing/                        # STATIC LANDING PAGE (scrollytelling)
│   ├── index.html
│   ├── main.css
│   ├── main.js
│   └── img/                        # Images / snapshots for the story
│
├── data/
│   ├── raw/                        # Raw data (scraped / downloaded, unmodified)
│   │   ├── records_100m_men_raw.csv
│   │   └── records_100m_women_raw.csv
│   │
│   ├── processed/                  # Cleaned + structured datasets
│   │   ├── records_100m_men.csv
│   │   └── records_100m_women.csv
│   │
│   └── predictions/                # Future projections from models
│       ├── 100m_model_pred_2100.csv
│       └── ...
│
├── notebooks/                      # Jupyter notebooks for EDA, modeling, checks
│   ├── 01_cleaning_100m.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_visual_checks.ipynb
│
├── src/                            # Reusable Python modules
│   ├── __init__.py
│   ├── cleaning.py                 # Data cleaning functions
│   ├── utils.py                    # Helpers (parsers, date handling, etc.)
│   ├── modeling.py                 # Training / loading prediction models
│   └── plotting.py                 # Plotly chart builders
│
└── docs/                           # Documentation (for the thesis / architecture)
    ├── architecture.md
    ├── data-dictionary.md
    └── roadmap.md
