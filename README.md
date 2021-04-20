# GloVe in R using Quanteda package 
Comparing word embeddings of terrorism text documents and non-terrorism text documents

# The data:
- Terrorism text document data = "only_tertext_data.csv"
- Non-terrorism text document data = "only_nontertext_data.csv"

Data was collected from a myriad of online sources, and the text documents obtained include online interviews, prison interviews, magazine articles, Tweets, and other samples of text (written and spoken) both from members of terrorist organizations and from members of non-terrorist organizations. 

Here, terrorist organizations refer to groups that (a) seek political, religious, economic, or other societal change and (b) use violence in pursuit of their goals. Non-terrorist organizations refer to groups that seek political, religious, economic, or other societal change but don’t use violence in pursuit of their goals.

Terrorists in the current sample include individuals affiliated with ISIS, Al-Qaeda, Al Shabaab, the Animal Liberation Front, the Ku Klux Klan, Skinheads, the Irish Republican Army (IRA), and the Deep Green Resistance. Non-terrorists in the current sample include individuals affiliated with the Society of the Muslim Brothers in Egypt, the Dalit Buddhist Movement, the Church of Scientology, the International Society for Krishna Consciousness, the National Rifle Association (NRA), the Camp for Climate Action, and the Extinction Rebellion. 

The current sample of data includes 151 text documents.


# RMD guide:
- Lines 27-70 = Uploading, cleaning, and preprocessing the data
- Lines 73-95 = Tokenizing the data 
- Lines 114-124 = Creating the feature co-occurence matrices
- Lines 128-146 = Training, fitting the GloVe models on each dataset
- Lines 153-167 = Generating the word vectors
- Lines 172-241 = Examining term representations/word vector similarity for common/frequent terms in both sets of data ("power", "fighting", and "movement")
- Lines 244-285 = Plotting the word embeddings (50) in 2d space for each dataset


# Resources I found helpful while working:
- https://quanteda.io/articles/pkgdown/replication/text2vec.html 
- https://github.com/prodriguezsosa/EmbeddingsPaperReplication
- http://text2vec.org/glove.html#glove_algorithm
- https://m-clark.github.io/text-analysis-with-R/word-embeddings.html
- https://smltar.com/embeddings.html
- https://rstudio-pubs-static.s3.amazonaws.com/132792_864e3813b0ec47cb95c7e1e2e2ad83e7.html
- https://smltar.com/mlclassification.html#classfirstattemptlookatdata


