# 🎧 SoundSense : Where Books Speak

## 📌 Project Description

SoundSense is an intelligent audiobook recommendation system that analyzes book
descriptions using Natural Language Processing (NLP) and cosine similarity to generate
personalized recommendations. The system applies TF-IDF vectorization and clustering
concepts with a Streamlit interface to deliver fast, accurate, and user-friendly results.

---

## 🎯 Objective

To develop a content-based audiobook recommendation system that uses text analysis
techniques and machine learning concepts to identify similar books and present
relevant suggestions through an interactive web interface.

---

## 🧩 Features

- ✅ Content-based recommendations using cosine similarity  
- ✅ NLP processing using TF-IDF vectorization  
- ✅ Clean and interactive Streamlit interface  
- ✅ Hidden Gems mode for discovering underrated books  
- ✅ Automatic column detection and data cleaning  
- ✅ User-controlled filtering using rating slider  

---
## 📂 Project Structure

      Audible Insights/
      │
      ├── app/
      │ └── app.py # Streamlit application
      │
      ├── data/
      │ ├── Audible_Catlog.csv
      │ └── Audible_Catlog_Advanced_Features.csv
      │
      ├── notebooks/
      │ └── audible_insights.ipynb
      │
      └── README.md 


---

## 🛠️ Technologies Used

- Python 🐍
- Scikit-learn
- Pandas, NumPy
- Streamlit
- Matplotlib & Seaborn
- Natural Language Processing (NLP)

---

## ⚙️ How It Works

1. Two datasets are loaded and merged using book name and author.
2. Missing values and duplicates are cleaned.
3. Book descriptions are converted into numeric vectors using TF-IDF.
4. Cosine similarity measures content similarity between books.
5. A Streamlit interface displays personalized recommendations.

---
## ⚠️ Limitations

      1. Does not use collaborative filtering.
      2. Genre-based filtering is disabled due to messy source data.
      3. Recommendations depend on quality of descriptions.
---

## 🚀 Future Enhancements

      1. Add user profiles and preferences
      2. Deploy on AWS / cloud platform
      3. Implement collaborative filtering
      4. Add dashboards and advanced analytics

---

