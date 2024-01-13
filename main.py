from flask import Flask, request, render_template
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        fig = plt.figure()
        text = request.form.get('text')
        stopWords = set(stopwords.words("english"))
         words = word_tokenize(text)
        num_sentences = 20
        # Create frequency table
        freq_table = dict()
        for word in words:
            word = word.lower()
            if word not in stop_words:
                if word in freq_table:
                    freq_table[word] += 1
                else:
                    freq_table[word] = 1
                    
        # Score sentences
        sentences = sent_tokenize(text)
        sentence_scores = dict()
        for sentence in sentences:
            for word in nltk.word_tokenize(sentence.lower()):
                if word in freq_table:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = freq_table[word]
                    else:
                        sentence_scores[sentence] += freq_table[word]
                        
        # Get highest scoring sentences
        summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        summary = ' '.join(summary_sentences)

        rouge_scores = rouge.get_scores(summary, text, avg = True)
        labels = list(rouge_scores.keys())
        values = [rouge_scores[label]['f'] for label in labels]  # get f-score for each label


        fig, axs = plt.subplots(3, figsize=(10, 15))

        for i, label in enumerate(rouge_scores.keys()):
            values = [rouge_scores[label][score_type] for score_type in rouge_scores[label].keys()]
            score_types = list(rouge_scores[label].keys())
            axs[i].scatter(score_types, values, color='blue')
            axs[i].set_xlabel('Score Types')
            axs[i].set_ylabel('Scores')
            axs[i].set_title(label)
            axs[i].grid(True)
        plt.tight_layout()
        return render_template('index.html', summary=summary, graph = fig)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host = "0.0.0.0")
