import pandas as pd
from transformers import pipeline
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import argparse
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


class WordCloudGenerator(PlotAnalyzer):
    def wordcloud(self, sub_df=None, text_col=None, max_words=100, title="Word Cloud"):
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        if sub_df is None:
            sub_df = self.df
        
        if text_col is None:
            text_col = self.text_col
        
        text = " ".join(sub_df[text_col].dropna().astype(str).tolist())
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=max_words
        ).generate(text)
        
        plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.imshow(wc)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


class CompositionChart(PlotAnalyzer):
    def plot(self):
        """Generate stacked bar chart showing dataset composition"""
        groups = {
            "Criminal Status": [
                ("Criminal", len(self.criminal_df)),
                ("Not Criminal", len(self.not_criminal_df)),
            ],
            "Expected Death": [
                ("Expected", len(self.expected_df)),
                ("Not Expected", len(self.not_expected_df)),
            ],
            "Religious Reference": [
                ("Religious", len(self.religion_df)),
                ("Not Religious", len(self.not_religion_df)),
            ],
            "By Century": [
                ("Pre-1700", len(self.pre_1700_df)),
                ("1700s", len(self.df_1700s)),
                ("1800s", len(self.df_1800s)),
                ("1900s", len(self.df_1900s)),
                ("2000s", len(self.df_2000s)),
            ],
        }
        
        plt.figure(figsize=(10, 5))
        x_pos = list(range(len(groups)))
        bottom = [0] * len(groups)
        
        for stack_i in range(max(len(v) for v in groups.values())):
            for bar_i, group_items in enumerate(groups.values()):
                if stack_i >= len(group_items):
                    continue
                label, value = group_items[stack_i]
                plt.bar(bar_i, value, bottom=bottom[bar_i])
                y_center = bottom[bar_i] + value / 2
                plt.text(bar_i, y_center, label, ha="center", va="center", fontsize=9)
                bottom[bar_i] += value
        
        plt.xticks(x_pos, list(groups.keys()), rotation=15)
        plt.title("Dataset Composition by Category")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        
class TopEmotionsChart(PlotAnalyzer):
    def plot(self, sub_df=None, title="Top 10 Emotions (Excluding Neutral)"):
        import matplotlib.pyplot as plt
        
        if sub_df is None:
            sub_df = self.df
        
        emotion_counts = (
            sub_df["emotion_label"]
            .loc[sub_df["emotion_label"].str.lower() != "neutral"]
            .value_counts()
            .head(10)
            .sort_values(ascending=True)
        )
        
        plt.figure(figsize=(10, 6))
        emotion_counts.plot(kind="barh")
        plt.title(title)
        plt.xlabel("Count")
        plt.ylabel("Emotion")
        plt.tight_layout()
        plt.show()


class EmotionHist(PlotAnalyzer):
    def plot(self, emotion_col, sub_df=None, title="Emotion Score Distribution", bins=10):
        import matplotlib.pyplot as plt
        
        if sub_df is None:
            sub_df = self.df
        
        plt.figure(figsize=(8, 5))
        plt.hist(sub_df[emotion_col].dropna(), bins=bins)
        plt.title(title)
        plt.xlabel(f"{emotion_col.replace('_', ' ').title()}")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Analysis and Visualization Pipeline")
    
    parser.add_argument("-f", "--file", required=True, help="Path to input CSV file")
    parser.add_argument("-t", "--text_col", required=True, help="Name of text column")
    parser.add_argument("-d", "--date_col", required=True, help="Name of date column")
    parser.add_argument("-o", "--output", help="Path to save processed CSV")
    parser.add_argument("-e", "--emotion", help="Specific emotion to analyze (e.g., remorse, joy)")
    parser.add_argument("-m", "--model", default="SamLowe/roberta-base-go_emotions", help="Emotion model name")
    parser.add_argument("--subset", choices=["criminal", "not_criminal", "expected", "not_expected", "religious", "not_religious", "pre_1700", "1700s", "1800s", "1900s", "2000s"], help="Subset of data to plot")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip emotion analysis (use existing columns)")
    parser.add_argument("--composition", action="store_true", help="Generate composition chart")
    parser.add_argument("--wordcloud", action="store_true", help="Generate word cloud")
    parser.add_argument("--top-emotions", action="store_true", help="Generate top emotions chart")
    parser.add_argument("--emotion-hist", action="store_true", help="Generate emotion histogram")
    parser.add_argument("--all-plots", action="store_true", help="Generate all visualizations")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    logging.info(f"Loading data from {args.file}")
    df = pd.read_csv(args.file)
    
    # Run emotion analysis
    if not args.skip_analysis:
        logging.info("Beginning emotion analysis")
        emotion_analyzer = EmotionAnalyzer(df, args.text_col, model_name=args.model)
        df = emotion_analyzer.apply_emotions()
        logging.info("Emotion analysis complete")
        
        if args.emotion:
            logging.info(f"Adding {args.emotion} scores")
            df = emotion_analyzer.apply_specific_emotion(args.emotion)
    
    # Helper to get subset df
    def get_subset_df(analyzer, subset_name):
        if subset_name is None:
            return analyzer.df
        subset_map = {
            "criminal": analyzer.criminal_df,
            "not_criminal": analyzer.not_criminal_df,
            "expected": analyzer.expected_df,
            "not_expected": analyzer.not_expected_df,
            "religious": analyzer.religion_df,
            "not_religious": analyzer.not_religion_df,
            "pre_1700": analyzer.pre_1700_df,
            "1700s": analyzer.df_1700s,
            "1800s": analyzer.df_1800s,
            "1900s": analyzer.df_1900s,
            "2000s": analyzer.df_2000s,
        }
        return subset_map[subset_name]
    
    # Generate visualizations
    if args.composition or args.all_plots:
        logging.info("Generating composition chart")
        comp_chart = CompositionChart(df, args.text_col, args.date_col)
        comp_chart.clean_year()
        comp_chart.clean_binary_columns()
        comp_chart.create_filtered_dfs()
        comp_chart.plot()
    
    if args.wordcloud or args.all_plots:
        logging.info("Generating word cloud")
        wc_gen = WordCloudGenerator(df, args.text_col, args.date_col)
        wc_gen.clean_year()
        wc_gen.clean_binary_columns()
        wc_gen.create_filtered_dfs()
        sub = get_subset_df(wc_gen, args.subset)
        title = f"Word Cloud - {args.subset}" if args.subset else "All Last Words"
        wc_gen.wordcloud(sub_df=sub, title=title)
    
    if args.top_emotions or args.all_plots:
        logging.info("Generating top emotions chart")
        emotions_chart = TopEmotionsChart(df, args.text_col, args.date_col)
        emotions_chart.clean_year()
        emotions_chart.clean_binary_columns()
        emotions_chart.create_filtered_dfs()
        sub = get_subset_df(emotions_chart, args.subset)
        title = f"Top Emotions - {args.subset}" if args.subset else "Top 10 Emotions (Excluding Neutral)"
        emotions_chart.plot(sub_df=sub, title=title)
    
    if args.emotion_hist or args.all_plots:
        emotion_col = f"{args.emotion}_score" if args.emotion else "emotion_score"
        logging.info(f"Generating histogram for {emotion_col}")
        hist_chart = EmotionHist(df, args.text_col, args.date_col)
        hist_chart.clean_year()
        hist_chart.clean_binary_columns()
        hist_chart.create_filtered_dfs()
        sub = get_subset_df(hist_chart, args.subset)
        title = f"{emotion_col} - {args.subset}" if args.subset else "Emotion Score Distribution"
        hist_chart.plot(emotion_col, sub_df=sub, title=title)
    
    # Save output
    if args.output:
        logging.info(f"Saving processed data to {args.output}")
        df.to_csv(args.output, index=False)
        logging.info("Done!")
    