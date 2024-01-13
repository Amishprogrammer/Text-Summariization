from flask import Flask, request, render_template
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get('text')
        stopWords = set(stopwords.words("english"))
         words = word_tokenize(text)
        
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
        return render_template('index.html', summary=summary)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
