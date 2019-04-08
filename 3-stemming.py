#to normalize your data just remove ing from ridinng to make to ride
#i was taking a ride in the car.
#i was riding the car.
# used to remove redundancy in the data set

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps=PorterStemmer()

example= ["pyhton","pythoner","pythoned","pythoned","pythonly"]
##
##for w in example:
##    print (ps.stem(w))

new_text="It is very important ot be pythonly while yuo are pythoning with python. All pythoners have pythoned at least once."

words= word_tokenize(new_text)

##
for w in words:
    print (ps.stem(w))
