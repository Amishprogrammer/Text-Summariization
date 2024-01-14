from flask import Flask, request, render_template
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
from rouge import Rouge
import matplotlib.image as mpimg


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        rouge = Rouge()
        fig = plt.figure()
        text = request.form.get('text')
        stop_words = set(stopwords.words("english"))
        words = word_tokenize(text)
        num_sentences = 6
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


        fig1, axs = plt.subplots(3, figsize=(10, 15))

        for i, label in enumerate(rouge_scores.keys()):
            values = [rouge_scores[label][score_type] for score_type in rouge_scores[label].keys()]
            score_types = list(rouge_scores[label].keys())
            axs[i].scatter(score_types, values, color='blue')
            axs[i].set_xlabel('Score Types')
            axs[i].set_ylabel('Scores')
            axs[i].set_title(label)
            axs[i].grid(True)
        plt.tight_layout()
        plt.savefig('my_fig.png')
        return render_template('index.html', rouge_1r = rouge_scores['rouge-1']['r'], rouge_1p = rouge_scores['rouge-1']['p'], rouge_1f = rouge_scores['rouge-1']['f'], rouge_2r = rouge_scores['rouge-2']['r'], rouge_2p = rouge_scores['rouge-2']['p'], rouge_2f= rouge_scores['rouge-2']['f'], rouge_lr = rouge_scores['rouge-l']['r'], rouge_lp = rouge_scores['rouge-l']['p'], rouge_lf= rouge_scores['rouge-l']['f'], summary=summary, graph = 'my_fig.png')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
