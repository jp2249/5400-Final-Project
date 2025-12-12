import pandas as pd
from transformers import pipeline
import numpy as np
import re

import logging

class EmotionAnalyzer:

    def __init__(self, df, text_col, model_name ="SamLowe/roberta-base-go_emotions", word_chunk_size =250):
        self.df = df
        self.text_col = text_col
        self.model_name = model_name
        self.word_chunk_size = word_chunk_size
        self.pipeline = pipeline("text-classification", model=self.model_name)
        self.pipeline_top_k = pipeline("text-classification", model=self.model_name, top_k=None)

    def chunk_text(self, text):
        """Chunk text into smaller pieces in case of long texts"""
        words = str(text).split()
        return [
            " ".join(words[i:i + self.word_chunk_size])
            for i in range(0, len(words), self.word_chunk_size)
        ]
    
    def get_top_emotion(self, text):
        """Get top emotion for text"""
        if text is None or str(text).strip() == "":
            return {"emotion_label": None, "emotion_score": np.nan}
        
        chunks = self.chunk_text(text)
        best_label = None
        best_score = None
        
        for chunk in chunks:
            result = self.pipeline(chunk)[0]
            score = float(result["score"])
            if best_score is None or score > best_score:
                best_label = result["label"]
                best_score = score
        
        return {
            "emotion_label": best_label,
            "emotion_score": round(best_score, 3) if best_score is not None else np.nan
        }

    def apply_emotions(self):
        """Get top emotion results and add to df"""
        rows = [self.get_top_emotion(t) for t in self.df[self.text_col]]
        emotion_df = pd.DataFrame(rows)
        self.df = pd.concat([self.df, emotion_df], axis=1)
        return self.df
    
    def get_specific_emotion_score(self, text, emotion_label):
        """Get score for specific emotion"""
        if text is None or str(text).strip() == "":
            return np.nan
        
        chunks = self.chunk_text(text)
        best_score = None
        
        for chunk in chunks:
            out = self.pipeline_top_k(chunk)
            if isinstance(out[0], dict):
                results = [out[0]]
            else:
                results = out[0]
            
            for r in results:
                if r["label"].lower() == emotion_label.lower():
                    score = float(r["score"])
                    if best_score is None or score > best_score:
                        best_score = score
        
        return round(best_score, 3) if best_score is not None else np.nan
    
    def apply_specific_emotion(self, emotion_label, column_name=None):
        """Get specific emotion results and add to df"""
        if column_name is None:
            column_name = f"{emotion_label}_score"
        
        self.df[column_name] = self.df[self.text_col].apply(
            lambda x: self.get_specific_emotion_score(x, emotion_label)
        )
        return self.df


class PlotAnalyzer:

    def __init__(self, df, text_col, date_col):
        self.df = df
        self.text_col = text_col
        self.date_col = date_col

    def clean_year(self):
        """Extract year from date column"""
        def extract_year(raw):
            if raw is None:
                return np.nan
            s = str(raw).lower().strip()
            if s == "":
                return np.nan
            s = re.sub(r"\[.*?\]|\(.*?\)", "", s)
            s = s.replace("o.s.", "").replace("c.", "").replace("ca.", "").replace("circa", "")
            s = s.replace("?", "").strip()
            century = re.search(r"(\d+)(st|nd|rd|th) century", s)
            if century:
                c = int(century.group(1))
                return -(c - 1) * 100 if "bc" in s else (c - 1) * 100
            if "bc" in s:
                m = re.search(r"\d{1,4}", s)
                return -int(m.group()) if m else np.nan
            s = s.replace("ad", "").strip()
            m4 = re.findall(r"\b\d{4}\b", s)
            if m4:
                return int(m4[-1])
            m2 = re.search(r"\b(\d{2})\b$", s)
            if m2:
                y = int(m2.group(1))
                return 2000 + y if y <= 25 else 1900 + y
            return np.nan
        
        def extract_year_row(row):
            for col in ["date", "title", "context"]:
                y = extract_year(row[col])
                if not pd.isna(y):
                    return y
            return np.nan
        
        self.df["year"] = self.df.apply(extract_year_row, axis=1)
        self.year_col = self.df["year"]
        return self.df
    
    def clean_binary_columns(self):
        """Convert binary columns to numeric"""
        self.df["is_expected"] = pd.to_numeric(self.df["is_expected"], errors="coerce")
        self.df["is_criminal"] = pd.to_numeric(self.df["is_criminal"], errors="coerce")
        self.df["is_religious"] = pd.to_numeric(self.df["is_religious"], errors="coerce")
        return self.df
    
    def filter_by_category(self, category, value):
        """Filter dataframe by category column value"""
        return self.df[self.df[category] == value]
    
    def filter_by_year_range(self, start_year=None, end_year=None):
        """Filter dataframe by year range"""
        if start_year is None and end_year is None:
            return self.df
        elif start_year is None:
            return self.df[self.df["year"] < end_year]
        elif end_year is None:
            return self.df[self.df["year"] >= start_year]
        else:
            return self.df[(self.df["year"] >= start_year) & (self.df["year"] < end_year)]
    
    def create_filtered_dfs(self):
        """Create all common filtered dataframes and store as attributes"""
        self.criminal_df = self.filter_by_category('is_criminal', 1)
        self.not_criminal_df = self.filter_by_category('is_criminal', 0)
        self.expected_df = self.filter_by_category('is_expected', 1)
        self.not_expected_df = self.filter_by_category('is_expected', 0)
        self.religion_df = self.filter_by_category('is_religious', 1)
        self.not_religion_df = self.filter_by_category('is_religious', 0)
        self.pre_1700_df = self.filter_by_year_range(end_year=1700)
        self.df_1700s = self.filter_by_year_range(1700, 1800)
        self.df_1800s = self.filter_by_year_range(1800, 1900)
        self.df_1900s = self.filter_by_year_range(1900, 2000)
        self.df_2000s = self.filter_by_year_range(2000)
        return self
