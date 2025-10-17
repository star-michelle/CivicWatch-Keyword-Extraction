**CivicWatch Keyword Extraction** is a data processing and analysis tool designed to extract meaningful keywords from political social media posts (currently focused on **Twitter**).  
It uses both **TF-IDF** and **KeyBERT** to identify and compare recurring topics, helping researchers and students explore online political discourse.

This project was developed as part of a university research workflow to make large-scale text data easier to interpret.

---

## 🌟 Features

- 📥 Load tweets directly from `.json.gz` files (Twitter API format)
- 🔍 Automatically filter posts by a list of political user handles
- 🧮 Extract keywords using:
  - **TF-IDF** (classic frequency-based approach)
  - **KeyBERT** (transformer-based semantic extraction)
- 🧹 Cleans and processes text to remove noise and URLs
- 💾 Exports results to a structured CSV file for further analysis
- ⚙️ Batch-processing support for large datasets
