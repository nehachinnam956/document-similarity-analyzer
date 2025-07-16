import os
import string
import matplotlib.pyplot as plt
import nltk
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

class TextAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Similarity Analyzer")
        self.root.geometry("800x600")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Folder selection
        ttk.Button(self.main_frame, text="Select Folder", command=self.select_folder).grid(row=0, column=0, pady=5)
        self.folder_label = ttk.Label(self.main_frame, text="No folder selected")
        self.folder_label.grid(row=0, column=1, pady=5)
        
        # Analysis options
        self.analysis_frame = ttk.LabelFrame(self.main_frame, text="Analysis Options", padding="5")
        self.analysis_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Button(self.analysis_frame, text="Compare Documents", command=self.analyze_documents).grid(row=0, column=0, padx=5)
        ttk.Button(self.analysis_frame, text="Generate Word Cloud", command=self.generate_wordcloud).grid(row=0, column=1, padx=5)
        ttk.Button(self.analysis_frame, text="Word Frequency", command=self.show_word_frequency).grid(row=0, column=2, padx=5)
        ttk.Button(self.analysis_frame, text="Sentiment Analysis", command=self.analyze_sentiment).grid(row=0, column=3, padx=5)
        ttk.Button(self.analysis_frame, text="Document Stats", command=self.show_document_stats).grid(row=0, column=4, padx=5)
        
        # Results area
        self.results_text = scrolledtext.ScrolledText(self.main_frame, height=15)
        self.results_text.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.documents = {}
        self.processed_documents = {}

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.folder_label.config(text=folder_path)
            self.documents = self.load_documents(folder_path)
            if self.documents:
                self.processed_documents = {filename: self.preprocess_text(content) 
                                         for filename, content in self.documents.items()}
                self.results_text.insert(tk.END, f"Loaded {len(self.documents)} documents.\n")
            else:
                self.results_text.insert(tk.END, "No documents found in selected folder.\n")

    def load_documents(self, folder_path):
        documents = {}
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        documents[filename] = file.read()
                except Exception as e:
                    self.results_text.insert(tk.END, f"Error reading {filename}: {e}\n")
        return documents

    def preprocess_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        return " ".join(words)

    def analyze_documents(self):
        if not self.processed_documents:
            self.results_text.insert(tk.END, "Please select a folder with documents first.\n")
            return

        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(self.processed_documents.values())
        similarity_matrix = cosine_similarity(doc_vectors)
        
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Print similarity scores
        doc_names = list(self.processed_documents.keys())
        self.results_text.insert(tk.END, "Document Similarity Scores:\n")
        for i in range(len(doc_names)):
            for j in range(i + 1, len(doc_names)):
                score = similarity_matrix[i][j]
                self.results_text.insert(tk.END, 
                    f"🔹 Similarity between '{doc_names[i]}' and '{doc_names[j]}': {score:.2f}\n")
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   xticklabels=doc_names, 
                   yticklabels=doc_names, 
                   annot=True, 
                   cmap='coolwarm',
                   fmt='.2f')
        plt.title("Document Similarity Heatmap")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    def show_word_frequency(self):
        if not self.processed_documents:
            self.results_text.insert(tk.END, "Please select a folder with documents first.\n")
            return
        
        # Get word frequencies across all documents
        all_words = " ".join(self.processed_documents.values()).split()
        word_freq = Counter(all_words).most_common(20)
        
        # Display results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Top 20 Most Frequent Words:\n\n")
        for word, freq in word_freq:
            self.results_text.insert(tk.END, f"📊 {word}: {freq} occurrences\n")
        
        # Plot bar chart
        plt.figure(figsize=(10, 6))
        words, freqs = zip(*word_freq)
        plt.bar(words, freqs)
        plt.xticks(rotation=45)
        plt.title("Top 20 Word Frequencies")
        plt.xlabel("Words")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def analyze_sentiment(self):
        if not self.documents:
            self.results_text.insert(tk.END, "Please select a folder with documents first.\n")
            return
        
        sia = SentimentIntensityAnalyzer()
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Sentiment Analysis Results:\n\n")
        
        for filename, content in self.documents.items():
            scores = sia.polarity_scores(content)
            sentiment = "positive" if scores['compound'] > 0 else "negative" if scores['compound'] < 0 else "neutral"
            self.results_text.insert(tk.END, f"📄 {filename}:\n")
            self.results_text.insert(tk.END, f"   Sentiment: {sentiment}\n")
            self.results_text.insert(tk.END, f"   Compound Score: {scores['compound']:.2f}\n")
            self.results_text.insert(tk.END, f"   Positive: {scores['pos']:.2f}\n")
            self.results_text.insert(tk.END, f"   Negative: {scores['neg']:.2f}\n")
            self.results_text.insert(tk.END, f"   Neutral: {scores['neu']:.2f}\n\n")

    def show_document_stats(self):
        if not self.documents:
            self.results_text.insert(tk.END, "Please select a folder with documents first.\n")
            return
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Document Statistics:\n\n")
        
        for filename, content in self.documents.items():
            words = content.split()
            sentences = content.split('.')
            chars = len(content)
            
            self.results_text.insert(tk.END, f"📄 {filename}:\n")
            self.results_text.insert(tk.END, f"   Word Count: {len(words)}\n")
            self.results_text.insert(tk.END, f"   Sentence Count: {len(sentences)}\n")
            self.results_text.insert(tk.END, f"   Character Count: {chars}\n")
            self.results_text.insert(tk.END, f"   Average Word Length: {chars/len(words):.2f}\n")
            self.results_text.insert(tk.END, f"   Average Sentence Length: {len(words)/len(sentences):.2f} words\n\n")

    def generate_wordcloud(self):
        if not self.documents:
            self.results_text.insert(tk.END, "Please select a folder with documents first.\n")
            return
            
        # Combine all documents
        combined_text = " ".join(self.processed_documents.values())
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            max_words=100).generate(combined_text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Word Cloud of All Documents")
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = TextAnalyzerGUI(root)
    root.mainloop()
