import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score,classification_report
from PIL import Image

import nltk
# import contractions
import inflect
from bs4 import BeautifulSoup
import re, string, unicodedata
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud 

from nltk.stem import LancasterStemmer, WordNetLemmatizer
from collections import defaultdict

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
from wordcloud import WordCloud, STOPWORDS
from collections import defaultdict
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# import server.maxMessageSize

def main():
    st.title("Navigating Data and ML Performance")
    st.sidebar.title("AccuProbe App")
    st.markdown("Sentiment Analysis")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    image = Image.open('emoji.jpg')
    st.image(image,'emojis')

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv(r'C:\\Users\\User\\Desktop\\glide_project_1\\tweet_emotions.csv')
        
        return data
    
    @st.cache_data(persist=True)
    def eda(df, row_limit=5, list_elements_limit=10):
        ## to print number of rows and columns
        df1 = pd.DataFrame(
             [
                  {" Rows and Columns":"Total number of rows in the dataset","Count":df.shape[0]}, 
                   {" Rows and Columns":"Total number of columns in the dataset","Count":df.shape[1]}
             ]
        )
        st.dataframe(df1,use_container_width=True)

        total_df = pd.DataFrame(df.dtypes).reset_index().rename(columns={0:'dtype', 'index':'column_name'})
        cat_df = total_df[total_df['dtype']=='object']
        num_df = total_df[total_df['dtype']!='object']
        st.write("\ninformation about type of data in different columns")
        df2 = pd.DataFrame(
             [
                  {" types of data":"Total number of categorical columns","Count":len(cat_df)}, 
                   {" types of data":"Total number of numerical columns","Count":len(total_df)-len(cat_df)}
             ]
        )
        st.dataframe(df2,use_container_width=True)
        

        #total_df['dtype'].value_counts().plot.bar()
        st.write(total_df.head(row_limit))

        print("==================================================")
        st.write("\nDescription of numerical column")

        #### Describibg numerical columns
        desc_df_num = df[list(num_df['column_name'])].describe().T.reset_index().rename(columns={'index':'column_name'})
        st.write(desc_df_num.head(row_limit))

        print("==================================================")
        st.write("\nDescription of categorical columns")

        desc_df_cat = df[list(cat_df['column_name'])].describe().T.reset_index().rename(columns={'index':'column_name'})
        st.write(desc_df_cat.head(row_limit))

        print("==================================================")
        st.write("\nTo check null values")

        df_isnull = df.isnull().sum()
        st.write(df_isnull)

        return
    
    @st.cache_data(persist=True)
    def countall(df):

        col = 'sentiment'
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18,8))

        # Assuming 'df' contains the DataFrame with the correct 'sentiment' column
        # Data preprocessing (if needed) should be done before this step

        # For the bar plot
        sns.countplot(data=df,y=col,order=df['sentiment'].value_counts().index,ax=ax1)
        ax1.set_title("Count of each emotion")

        # For the donut plot
        sizes = df[col].value_counts()[:10]
        labels = sizes.index.tolist()  # Get the sentiment categories from the value counts
        explode = [0.1] * len(sizes)   # Customize the explode values if needed
        ax2.pie(sizes, explode=explode, startangle=45, labels=labels, autopct='%1.0f%%', pctdistance=0.85)
        ax2.add_artist(plt.Circle((0, 0), 0.7, fc='white'))  # Customize the circle size if needed
        ax2.set_title("Percentage of each emotion")

        plt.tight_layout()
        st.pyplot(fig)


        st.markdown(
                """
                Observations:
                - Here the distribution of data in each category shows that the data is unbalanced ,
                - In order to make the dataset balanced (for good prediction) , we keep the first five dataset class as a seperate five category and the rest of the data in one category (mixed).

                """
                )
        df['sentiment'] = df['sentiment'].apply(lambda x : x if x in ['happiness', 'sadness', 'worry', 'neutral', 'love'] else "mixed")
        col = 'sentiment'
        fig, (ax1, ax2)  = plt.subplots(nrows=1, ncols=2, figsize=(18,8))
        explode = list((np.array(list(df[col].dropna().value_counts()))/sum(list(df[col].dropna().value_counts())))[::-1])[:10]
        labels = list(df[col].dropna().unique())[:10]
        sizes = df[col].value_counts()[:10]
        
        ax2.pie(sizes,  explode=explode, startangle=60, labels=labels,autopct='%1.0f%%', pctdistance=0.9)
        ax2.add_artist(plt.Circle((0,0),0.6,fc='white'))
        sns.countplot(y =col, data = df, ax=ax1)
        ax1.set_title("Count of each emotion")
        ax2.set_title("Percentage of each emotion")
        st.pyplot(fig)

        df['char_length'] = df['content'].apply(lambda x : len(x))
        df['token_length'] = df['content'].apply(lambda x : len(x.split(" ")))

        fig, (ax1, ax2)  = plt.subplots(nrows=1, ncols=2, figsize=(18,8))
        sns.distplot(df['char_length'],color="green", ax=ax1)
        sns.distplot(df['token_length'],color="green", ax=ax2)
        ax1.set_title('Number of characters in the tweet')
        ax2.set_title('Number of token(words) in the tweet')
        st.pyplot(fig)


        st.markdown('Distribution of character length sentiment wise')
        fig, ax = plt.subplots(figsize=(12,6))
        for sentiment in df['sentiment'].value_counts().sort_values()[-5:].index.tolist():
            sns.kdeplot(df[df['sentiment']==sentiment]['char_length'],ax=ax, label=sentiment)
        ax.legend()
        ax.set_title("Distribution of character length sentiment-wise [Top 5 sentiments]")
        st.pyplot(fig)
        
        st.markdown('Distribution of token length sentiment wise')
        fig, ax = plt.subplots(figsize=(12,6))
        for sentiment in df['sentiment'].value_counts().sort_values()[-5:].index.tolist():
            #print(sentiment)
            sns.kdeplot(df[df['sentiment']==sentiment]['token_length'],ax=ax, label=sentiment)
        ax.legend()
        ax.set_title("Distribution of token length sentiment-wise [Top 5 sentiments]")
        st.pyplot(fig)

        avg_df = df.groupby('sentiment').agg({'char_length':'mean', 'token_length':'mean'})
        fig, (ax1, ax2)  = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
        ax1.bar(avg_df.index, avg_df['char_length'],color="orange")
        ax2.bar(avg_df.index, avg_df['token_length'], color='green')
        ax1.set_title('Avg number of characters')
        ax2.set_title('Avg number of token(words)')
        ax1.set_xticklabels(avg_df.index, rotation = 45)
        ax2.set_xticklabels(avg_df.index, rotation = 45)
        st.pyplot(fig)
        return
    
    @st.cache_data(persist=True)
    def text_preprocessing_platform(df, text_col, remove_stopwords=True):

        ## Define functions for individual steps
        # First function is used to denoise text
        def denoise_text(text):
            # Strip html if any. For ex. removing <html>, <p> tags
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text()
            # Replace contractions in the text. For ex. didn't -> did not
            contractions = {
                "ain't": "am not / are not",
                "aren't": "are not / am not",
                "can't": "cannot",
                "can't've": "cannot have",
                "'cause": "because",
                "could've": "could have",
                "couldn't": "could not",
                "couldn't've": "could not have",
                "didn't": "did not",
                "doesn't": "does not",
                "don't": "do not",
                "hadn't": "had not",
                "hadn't've": "had not have",
                "hasn't": "has not",
                "haven't": "have not",
                "he'd": "he had / he would",
                "he'd've": "he would have",
                "he'll": "he shall / he will",
                "he'll've": "he shall have / he will have",
                "he's": "he has / he is",
                "how'd": "how did",
                "how'd'y": "how do you",
                "how'll": "how will",
                "how's": "how has / how is",
                "i'd": "I had / I would",
                "i'd've": "I would have",
                "i'll": "I shall / I will",
                "i'll've": "I shall have / I will have",
                "i'm": "I am",
                "i've": "I have",
                "isn't": "is not",
                "it'd": "it had / it would",
                "it'd've": "it would have",
                "it'll": "it shall / it will",
                "it'll've": "it shall have / it will have",
                "it's": "it has / it is",
                "let's": "let us",
                "ma'am": "madam",
                "mayn't": "may not",
                "might've": "might have",
                "mightn't": "might not",
                "mightn't've": "might not have",
                "must've": "must have",
                "mustn't": "must not",
                "mustn't've": "must not have",
                "needn't": "need not",
                "needn't've": "need not have",
                "o'clock": "of the clock",
                "oughtn't": "ought not",
                "oughtn't've": "ought not have",
                "shan't": "shall not",
                "sha'n't": "shall not",
                "shan't've": "shall not have",
                "she'd": "she had / she would",
                "she'd've": "she would have",
                "she'll": "she shall / she will",
                "she'll've": "she shall have / she will have",
                "she's": "she has / she is",
                "should've": "should have",
                "shouldn't": "should not",
                "shouldn't've": "should not have",
                "so've": "so have",
                "so's": "so as / so is",
                "that'd": "that would / that had",
                "that'd've": "that would have",
                "that's": "that has / that is",
                "there'd": "there had / there would",
                "there'd've": "there would have",
                "there's": "there has / there is",
                "they'd": "they had / they would",
                "they'd've": "they would have",
                "they'll": "they shall / they will",
                "they'll've": "they shall have / they will have",
                "they're": "they are",
                "they've": "they have",
                "to've": "to have",
                "wasn't": "was not",
                "we'd": "we had / we would",
                "we'd've": "we would have",
                "we'll": "we will",
                "we'll've": "we will have",
                "we're": "we are",
                "we've": "we have",
                "weren't": "were not",
                "what'll": "what shall / what will",
                "what'll've": "what shall have / what will have",
                "what're": "what are",
                "what's": "what has / what is",
                "what've": "what have",
                "when's": "when has / when is",
                "when've": "when have",
                "where'd": "where did",
                "where's": "where has / where is",
                "where've": "where have",
                "who'll": "who shall / who will",
                "who'll've": "who shall have / who will have",
                "who's": "who has / who is",
                "who've": "who have",
                "why's": "why has / why is",
                "why've": "why have",
                "will've": "will have",
                "won't": "will not",
                "won't've": "will not have",
                "would've": "would have",
                "wouldn't": "would not",
                "wouldn't've": "would not have",
                "y'all": "you all",
                "y'all'd": "you all would",
                "y'all'd've": "you all would have",
                "y'all're": "you all are",
                "y'all've": "you all have",
                "you'd": "you had / you would",
                "you'd've": "you would have",
                "you'll": "you shall / you will",
                "you'll've": "you shall have / you will have",
                "you're": "you are",
                "you've": "you have"
                }
            for word in text.split():
                if word.lower() in contractions:
                    text = text.replace(word, contractions[word.lower()])
            
            # text = contractions.fix(text)
            return text

        ## Next step is text-normalization

        # Text normalization includes many steps.

        # Each function below serves a step.


        def remove_non_ascii(words):
            """Remove non-ASCII characters from list of tokenized words"""
            new_words = []
            for word in words:
                new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
                new_words.append(new_word)
            return new_words


        def to_lowercase(words):
            """Convert all characters to lowercase from list of tokenized words"""
            new_words = []
            for word in words:
                new_word = word.lower()
                new_words.append(new_word)
            return new_words


        def remove_punctuation(words):
            """Remove punctuation from list of tokenized words"""
            new_words = []
            for word in words:
                new_word = re.sub(r'[^\w\s]', '', word)
                if new_word != '':
                    new_words.append(new_word)
            return new_words


        def replace_numbers(words):
            """Replace all interger occurrences in list of tokenized words with textual representation"""
            # p = inflect.engine()
            p = inflect.engine()
            new_words = []
            for word in words:
                if word.isdigit():
                    new_word = p.number_to_words(word)
                    new_words.append(new_word)
                else:
                    new_words.append(word)
            return new_words


        def remove_stopwords(words):
            """Remove stop words from list of tokenized words"""
            new_words = []
            for word in words:
                if word not in stopwords.words('english'):
                    new_words.append(word)
            return new_words


        def stem_words(words):
            """Stem words in list of tokenized words"""
            stemmer = LancasterStemmer()
            stems = []
            for word in words:
                stem = stemmer.stem(word)
                stems.append(stem)
            return stems


        def lemmatize_verbs(words):
            """Lemmatize verbs in list of tokenized words"""
            lemmatizer = WordNetLemmatizer()
            lemmas = []
            for word in words:
                lemma = lemmatizer.lemmatize(word, pos='v')
                lemmas.append(lemma)
            return lemmas


        ### A wrap-up function for normalization
        def normalize_text(words, remove_stopwords):
            words = remove_non_ascii(words)
            words = to_lowercase(words)
            words = remove_punctuation(words)
            words = replace_numbers(words)
            if remove_stopwords:
                words = remove_stopwords(words)
            #words = stem_words(words)
            words = lemmatize_verbs(words)
            nltk.download('wordnet')
            return words

        # All above functions work on word tokens we need a tokenizer

        # Tokenize tweet into words
        def tokenize(text):
            return nltk.word_tokenize(text)


        # A overall wrap-up function
        def text_prepare(text):
            text = denoise_text(text)
            text = ' '.join([x for x in normalize_text(tokenize(text), remove_stopwords)])
            return text

        # run every-step
        df[text_col] = [text_prepare(x) for x in df[text_col]]


        # return processed df
        return df
        

    
    @st.cache_data(persist=True)
    def split(df):
        # select X and y for model building
        x = df['content']
        y = df['sentiment']
        Lbe = LabelEncoder()
        y =  Lbe.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

        return X_train, X_test, y_train, y_test
    
    def print_word_cloud(df, sentiment):

        print("Word cloud of most frequent words for the sentiment : {}".format(sentiment))

        temp_df = df[df['sentiment']==sentiment]
        print("Number of Rows : ", len(temp_df))

        corpus = ''
        for text in temp_df.content:
            text = str(text)
            corpus += text

        total = 0
        count = defaultdict(lambda: 0)
        for word in corpus.split(" "):
            total += 1
            count[word] += 1

        top20pairs = sorted(count.items(), key=lambda kv: kv[1], reverse=True)[:20]
        top20words = [i[0] for i in top20pairs]
        top20freq = [i[1] for i in top20pairs]

        xs = np.arange(len(top20words))
        width = 0.5

        fig = plt.figure(figsize=(10,6))
        ax = fig.gca()  #get current axes
        ax.bar(xs, top20freq, width, align='center',color = 'green')

        ax.set_xticks(xs)
        ax.set_xticklabels(top20words)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # lower max_font_size, change the maximum number of word and lighten the background:
        wordcloud = WordCloud(max_font_size=50, max_words=50,stopwords=stop_words, background_color="white").generate(corpus)
        fig = plt.figure(figsize = (10, 6), facecolor = None)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(fig)

    
    
    df = load_data()

    st.sidebar.subheader('Analyze Data')
    Analyzer = st.sidebar.selectbox("Analyzer", ("Basic EDA", "Categorical Data Analysis","Text Preprocessing", "Most frequent words"))
    if Analyzer == 'Basic EDA':
        if st.sidebar.button("Analyze", key='Analyze'):
            st.subheader("Basic EDA")
            eda(df)
    if Analyzer == 'Categorical Data Analysis':
        if st.sidebar.button("Analyze", key='Analyze'):
            st.subheader("Categorical Data Analysis Results")
            countall(df)
    dfp = df
    df = text_preprocessing_platform(df, 'content', remove_stopwords=False)
    if Analyzer == 'Text Preprocessing':
        if st.sidebar.button("Analyze", key='Analyze'):
            st.subheader("Text Preprocessing")
            st.write("Before Text Preprocessing")
            a1 = dfp.head()[['content']]
            st.write(a1)
            # processed_df = text_preprocessing_platform(df, 'content', remove_stopwords=False)
            # df = text_preprocessing_platform(df, 'content', remove_stopwords=False)
            st.write("After Text Preprocessing")
            st.write(df.head()[['content']])
            

    if Analyzer == 'Most frequent words':
        if st.sidebar.button("Analyze", key='Analyze'):
            st.subheader("Most frequent occured words in each category")
            st.write("### Most frequent occured words in sadness")
            print_word_cloud(df, 'sadness')
            st.write("### Most frequent occured words in happiness")
            print_word_cloud(df, 'happiness')
            st.write("### Most frequent occured words in neutral")
            print_word_cloud(df, 'neutral')
            st.write("### Most frequent occured words in worry")
            print_word_cloud(df, 'worry')
            st.write("### Most frequent occured words in love")
            print_word_cloud(df, 'love')

        
    
    df['sentiment'] = df['sentiment'].apply(lambda x : x if x in ['happiness', 'sadness', 'worry', 'neutral', 'love'] else "mixed")
    x_train, x_test, y_train, y_test = split(df)
    tf = TfidfVectorizer(analyzer='word',max_features=1000,ngram_range=(1,2))
    x_tf = tf.fit_transform(x_train) 
    x_val_tf = tf.transform(x_test) 

    x_tf   = x_tf.toarray() 
    x_val_tf =  x_val_tf.toarray()
    


    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("MultiNomial Naive Bayes", "Logistic Regression", "Random Forest"))

    if classifier == 'MultiNomial Naive Bayes':

        model =MultinomialNB()
        model.fit(x_tf,y_train)
        pred1 = model.predict(x_val_tf)

        
        if st.sidebar.button("Classify", key='classify'):
            st.write('Classification Report (Naive Bayes)')
            
            report = classification_report(y_test,pred1,output_dict=True)
            df3 = pd.DataFrame(report).transpose()
            df3
            
    
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        #set parameters  
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')
        solver = st.sidebar.radio("Solver", ("saga", "liblinear", "newton-cg","lbfgs"), key='solver')
        pnlty = st.sidebar.radio("penality",("l1","l2"),key='pnlty')


        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model =LogisticRegression(solver=solver,C=C,penalty=pnlty,max_iter=max_iter)
            model.fit(x_tf,y_train)
            pred1 = model.predict(x_val_tf)
            report = classification_report(y_test,pred1,output_dict=True)
            df3 = pd.DataFrame(report).transpose()
            df3
    
    if classifier == 'Random Forest':
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model =RandomForestClassifier()
            model.fit(x_tf,y_train)
            pred1 = model.predict(x_val_tf)
            report = classification_report(y_test,pred1,output_dict=True)
            df3 = pd.DataFrame(report).transpose()
            df3
    
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Heart Attack Prediction Data Set (Classification)")
        st.write(df)
        st.markdown("This [data set](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset) includes various attributes , categorical (such as sex,chest pain type ,fasting blood sugar level etc) and nummerical (such as age,cholestrol, oldpeak etc) that will predict the risk of heart attack  ) ) "
        "Heart attack is a medical emergency and is very common with approximately 10 million case per year (in india))"
        "It occurs when the flow of the blood to the heart is sevverely  reduced  or blocked")

        st.subheader("Data Variable Description")
        st.markdown(" 1. age -age in years")
        st.markdown("2. sex - (1 = male, 0 = female) ")
        st.markdown(
            """
            3. cp - chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 0 = asymptomatic):
                - typical angina : symptoms are chest, arm, jaw pain
                - atypical angina : symptoms are epigastric , back pain, burning, stabbing
                - non anginal pain : people having chest pain without heart disease. they also suffer from panic disorder, anxiety, depression
            """
            )
        st.markdown("4. trestbps - resting blood pressure (in mm Hg on admission to the hospital)")
        st.markdown("5. chol - serum cholestoral in mg/dl")
        st.markdown(
            """
            6. fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false):
                - it indicates blood sugar level after an overnight fast.
                - fbs < 120mg/dL is normal and fbs > 120 indicates patient have diabetes
            """
            )
        st.markdown(
            """
            7. rest_ecg - resting electrocardiographic results (1 = normal; 2 = having ST-T wave abnormality; 0 = hypertrophy):
                - hypertrophy : inndicates the thickening of wall of heart's main pumping chamber.
                - having ST-T wave abnormality : indicates the blockage of the main artery.
            """
            )
        st.markdown("8. thalach - maximum heart rate achieved")
        st.markdown(
            """
            9. exang - exercise induced angina (1 = yes; 0 = no):
                - exercise induced angina are temporary and goes away with rest.
            """
            )
        st.markdown("10. oldpeak - ST depression induced by exercise relative to rest")
        st.markdown("11. slope - the slope of the peak exercise ST segment (2 = upsloping; 1 = flat; 0 = downsloping)")
        st.markdown(
            """
            12. ca - number of major vessels (0-3) colored by flourosopy
                - major vessels are the blood vessels that are directly connected to heart.
                - eg : pulmonary artery, pulmonary veins , vena cava, aorta
            """
            )
        st.markdown(
            """
            13. thal - (2 = normal; 1 = fixed defect; 3 = reversable defect) :
                - Blood disorder called thalassemia.
                - normal: indicates normal blood flow.
                - fixed defects : no blood flow in some parts of the heart.
                - reversible defects : blood flow is observed but it is not normal.
            """
            )
        st.markdown("14. num - the predicted attribute - diagnosis of heart disease (angiographic disease status) (Value 0 = < diameter narrowing; Value 1 = > 50% diameter narrowing)")
        st.markdown("The reference for the above description are taken from  [kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/discussion/234843) and [National Library of Medicine](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4000924/)")

        

if __name__ == '__main__':
    main()


