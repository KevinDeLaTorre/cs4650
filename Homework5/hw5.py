import spacy
from newsapi import NewsApiClient
import pickle
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nlp_eng = spacy.load('en_core_web_lg')
newsapi = NewsApiClient(api_key='68add3f8e2c24d918ace247a0da639f1')

print( "Getting articles..." )
articles = []
for i in range( 1, 6 ):
  temp = newsapi.get_everything(q='coronavirus', language='en', from_param='2021-03-01', to='2020-03-26', sort_by='relevancy', page=i)
  articles.append( temp )

print( "Done.")

filename = 'articlesCovid.pckl'
pickle.dump(articles, open(filename, 'wb' ) )

filename = 'articlesCovid.pckl'
loaded_model = pickle.load( open( filename, 'rb' ) )

filepath = './articlesCovid.pckl'
pickle.dump( loaded_model, open( filepath, 'wb' ) )

dados = []
for i, article in enumerate( articles ):
  for x in article[ 'articles' ]:
    title = x[ 'title' ]
    description = x[ 'description' ]
    content = x[ 'content' ]
    dados.append( {'title':title, 'date':x[ 'publishedAt' ], 'desc':description, 'content':content } )
  
  df = pd.DataFrame( dados )
  df = df.dropna()

  pos_tag = ['PROPN', 'VERB', 'NOUN' ]

def get_keywords_eng( text ):
  nlp = spacy.load( "en_core_web_lg" )
  doc = nlp( text )
  result = []
  for token in doc:
    if ( token.text in nlp_eng.Defaults.stop_words or token.pos_ == 'PUNCT' ):
      continue
    if (token.pos_ in pos_tag ):
      result.append( token.text )
  
  return result

print( "Processing keywords..." )
results = []
for content in df.content.values:
  results.append( [ ( '#' + x[0] ) for x in Counter( get_keywords_eng( content ) ).most_common( 5 ) ] )
print( "Done" )

df[ 'keywords' ] = results

print( df[ 'keywords' ] )
print( "Saving to keywords.csv" )
df[ 'keywords' ].to_csv("./keywords.csv", header=None, index=None)
print( "Done saving." )

print( "Generating wordcloud..." )
text = str( results )
wordcloud = WordCloud( max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()