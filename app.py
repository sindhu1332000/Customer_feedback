from flask import Flask, render_template,url_for,redirect,request
from flask_mysqldb import MySQL
import nltk  
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re, string
from nltk.corpus import stopwords
from nltk import FreqDist
import random
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/checkout')
def checkout():
    return render_template('checkout.html')

@app.route('/my-account')
def myaccount():
    return render_template('my-account.html')

@app.route('/shop')
def shop():
    return render_template('shop.html')

@app.route('/wishlist')
def wishlist():
    return render_template('wishlist.html')

@app.route('/service', methods=['GET', 'POST'])
def service():
    if request.method == 'POST':
        # do stuff when the form is submitted

        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('index'))

    # show the form, it wasn't submitted
    return render_template('service.html')



@app.route('/contactus', methods=['GET', 'POST'])
def contactus():       
    if request.method == 'POST':
        # do stuff when the form is submitted
        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('index'))

    # show the form, it wasn't submitted
    return render_template('contactus.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

# settings
app.secret_key = "mysecretkey"

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'mydb'

mysql = MySQL(app)

@app.route('/handle_data', methods=['POST'])
def handle_data():
    name = request.form['name']
    email = request.form['email']
    product = request.form['product']
    message = request.form['message']
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO user_feedback(name, email, product, message) VALUES (%s, %s, %s, %s)", (name, email, product, message))
    mysql.connection.commit()
    '''flask('Added successfully')'''
    cur.close()			
    print(name,email,product,message)
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    def lemmatize_sentence(tokens):
        lemmatizer = WordNetLemmatizer()
        lemmatized_sentence = []
        for word, tag in pos_tag(tokens):
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
        return lemmatized_sentence
    def remove_noise(tweet_tokens, stop_words = ()):
        cleaned_tokens = []
        for token, tag in pos_tag(tweet_tokens):
            token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                        '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
            token = re.sub("(@[A-Za-z0-9_]+)","", token)
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)
            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleaned_tokens.append(token.lower())
        return cleaned_tokens
    stop_words = stopwords.words('english')
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    def get_all_words(cleaned_tokens_list):
        for tokens in cleaned_tokens_list:
            for token in tokens:
                yield token
    all_pos_words = get_all_words(positive_cleaned_tokens_list)
    freq_dist_pos = FreqDist(all_pos_words)
    def get_tweets_for_model(cleaned_tokens_list):
        for tweet_tokens in cleaned_tokens_list:
            yield dict([token, True] for token in tweet_tokens)
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]
    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]
    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)
    train_data = dataset[:7000]
    test_data = dataset[7000:]
    classifier = NaiveBayesClassifier.train(train_data)
    custom_tweet = message
    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    ans=classifier.classify(dict([token, True] for token in custom_tokens))
    return render_template('contactus.html',ans=ans)
    def update_contact(review):
        if request.method == 'POST':
            name = request.form['name']
            email = request.form['email']
            product = request.form['product']
            message = request.form['message']
            cur = mysql.connection.cursor()
            cur.execute("""
                UPDATE contacts
                SET name = %s,
                    email = %s,
                    product = %s
                WHERE  review= %s
            """, (name, email, phone, review))
            flash('Contact Updated Successfully')
            mysql.connection.commit()
            return redirect(url_for('contactus'))


@app.route('/')
def Index():
    cur = mysql.connection.cursor()
    cur.execute('SELECT * FROM user_feedback')
    data = cur.fetchall()
    cur.close()
    return render_template('index.html', contacts = data)


        
if __name__ == '__main__':
    app.run(debug=True)