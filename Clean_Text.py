import string
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
import nltk

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

class CleanText:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def clean_text(self, text):
        # إزالة المعادلات الرياضية
        text = re.sub(r'\b[\w\s\+\-\*/\^\(\)]+=[\w\s\+\-\*/\^\(\)]+', '', text)
        # إزالة روابط HTML
        text = re.sub(r'http\S+', '', text)
        # إزالة التعبيرات غير الضرورية
        text = re.sub(r'\b(eww+|aww+|wow+|omg+|lol+)\b', '', text, flags=re.IGNORECASE)
        # إزالة الرموز الفردية غير المرغوب فيها
        text = re.sub(r'\b\w\b', '', text)
        # إزالة الجمل بين القوسين
        text = re.sub(r'\(.*?\)', '', text)
        # تحويل النص إلى أحرف صغيرة وإزالة علامات الترقيم
        lowercased = text.lower().translate(str.maketrans('', '', string.punctuation))
        # إزالة كلمات التوقف باستخدام Spacy
        doc = nlp(lowercased)
        no_stopwords = ' '.join([token.text for token in doc if not token.is_stop])
        # إزالة الأرقام
        no_numbers = re.sub(r'\d+', '', no_stopwords)
        # تقطيع النص إلى كلمات وتطبيق التلميع والتجذير
        words = word_tokenize(no_numbers)
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        stemmed_words = [stemmer.stem(word) for word in lemmatized_words]
        # إعادة تجميع الكلمات وإزالة المسافات الزائدة
        no_whitespace = ' '.join(stemmed_words).strip()
        return no_whitespace
