#similiar to stemming but ends up always giving better results in terms of the same word or its synonym
#better than stemming because it always return an existing word
#always pass the pos if the word is not a noun
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print (lemmatizer.lemmatize("cats"))
print (lemmatizer.lemmatize("better",pos="a"))
print (lemmatizer.lemmatize("geese"))
print (lemmatizer.lemmatize("python"))
print (lemmatizer.lemmatize("running",'v'))
print (lemmatizer.lemmatize("cooling","v"))
print (lemmatizer.lemmatize("better"))
print (lemmatizer.lemmatize("cacti","a"))
print (lemmatizer.lemmatize("cacti"))
