##########################################
# Amanda Moeller
# Methods tutorial
# Text embeddings with GloVe & quanteda 
# PLSC 597
# Spring 2021
# April 20, 2021
##########################################

# BACKGROUND: ####################################################################################

# In this tutorial, we:
#     1) compare word embeddings of terrorist and non-terrorist text documents
#     2) Create a Naive Bayes model that uses word embeddings to predict whether a document was written/spoken by a terrorist or a non-terrorist

# Our text docs (n=151) include online interviews, prison interviews, magazine articles, Tweets, and other text (written and spoken) 
# These documents originated from members of terrorist organizations and from members of non-terrorist organizations.

# Here, terrorist organizations refer to groups that 
#     (a) seek political, religious, economic, or other societal change, and 
#     (b) use violence in pursuit of their goals. 
# Non-terrorist organizations refer to groups that 
#     (a) seek political, religious, economic, or other societal change, but 
#     (b) don’t use violence in pursuit of their goals.

# So basically, we're trying to figure out whether we can use text embeddings from natural language to differentiate between terrorists and non-terrorists

# Library packages: ####################################################################################################################
#     install.packages("readr")
library(readr)
#     install.packages("quanteda")
library(quanteda)
#     install.packages("quanteda.textstats")
library(quanteda.textstats)
#     install.packages("tidytext")
library(tidytext)
#     install.packages("tm")
library(tm)
#     install.packages("text2vec")
library(text2vec)
#     install.packages("irlba")
library(irlba)
#     install.packages("tokenizers")
library(tokenizers)
#     install.packages("magrittr")
library(magrittr)
#     install.packages("psclr")
library(pscl)
#     install.packages("ggplot2")
library(ggplot2)

# Bring in the data ##########################################################################################################

urlfile=("https://raw.githubusercontent.com/anm332/GloVe-in-R-with-Quanteda-Word-Embeddings-of-Terrorism-Data-/main/only_tertext_data.csv")
ter_df<-read_csv(url(urlfile)) # here's the terrorism data
ter_df$rawtext <- ter_df$`Text (raw data)` # change name of text variable

urlfile=("https://raw.githubusercontent.com/anm332/GloVe-in-R-with-Quanteda-Word-Embeddings-of-Terrorism-Data-/main/only_nontertext_data.csv")
nonter_df<-read_csv(url(urlfile)) # here's the non-terrorism data


urlfile=("https://raw.githubusercontent.com/anm332/GloVe-in-R-with-Quanteda-Word-Embeddings-of-Terrorism-Data-/main/2021_latestdata.csv")
fulldf <- read_csv(url(urlfile)) # and here's the full dataframe (above two combined)
fulldf$rawtext <- fulldf$`Text (raw data)`


# tokenize & clean up both sets of data (normalization): #####################################################################

# terrorism tokens
ter_toks <- tokens(ter_df$rawtext, remove_punct = TRUE) # cleaning: remove punctuation
ter_toks <- tokens_remove(ter_toks, stopwords("english"))  # remove stopwords
ter_toks <- tokens(ter_toks, removeNumbers) # remove numbers
ter_toks <- tokens(ter_toks, removePunctuation) # remove punctuation
ter_toks <- tokens(ter_toks, stemDocument()) # stem words 
class(ter_toks)

# non-terrorism tokens
nonter_toks <- tokens(nonter_df$rawtext, remove_punct = TRUE)
nonter_toks <- tokens_remove(nonter_toks, stopwords("english")) 
nonter_toks <- tokens(nonter_toks, removeNumbers)
nonter_toks <- tokens(nonter_toks, removePunctuation)
nonter_toks <- tokens(nonter_toks, stemDocument())
class(nonter_toks)

# full df tokens
df_toks <- tokens(fulldf$rawtext, remove_punct = TRUE)
df_toks <- tokens_remove(df_toks, stopwords("english")) 
df_toks <- tokens(df_toks, removeNumbers)
df_toks <- tokens(df_toks, removePunctuation)
df_toks <- tokens(df_toks, stemDocument())
class(df_toks)

# Create the corpus and matrices for all the data: ############################################################
# (corpus = a collection of text)

# 1) terrorism data corpus & matrix
ter_corp <- corpus(ter_df, text_field="rawtext")
ter_dfm <- dfm(ter_corp) 
ter_dfm # here's the terrorism corpus

ter_corp_matrix <- as.matrix(ter_corp) # terrorism corpus as a matrix
ter_corp_df <- 
  data.frame(text = sapply(ter_corp, 
                           as.character), 
                          stringsAsFactors = FALSE) # terrorism corpus as a df

# 2) non-terrorism corpus, matrix
nonter_corp <- corpus(nonter_df, text_field="rawtext")
nonter_dfm <- dfm(nonter_corp, remove_punct = TRUE, remove = stopwords('en'))
nonter_dfm

nonter_corp_matrix <- as.matrix(nonter_corp)
nonter_corp_df <- data.frame(text = sapply(nonter_corp, as.character), stringsAsFactors = FALSE)


# 3) full dataframe corpus, matrix
corpus <- corpus(fulldf, text_field="rawtext")
full_dfm <- dfm(corpus, remove_punct = TRUE, remove = stopwords('en'))
full_dfm

full_corp_matrix <- as.matrix(corpus)
full_corp_df <- data.frame(text = sapply(corpus, as.character), stringsAsFactors = FALSE)


# From our tokens objects, get the names of the features that occur 5+ times ##################################################################
# (Here we're trimming the features before constructing the feature co-occurrence matrix): 

# terrorism data features:
t_feats <- dfm(ter_toks, verbose = TRUE) %>%
  dfm_trim(min_termfreq = 5) %>%
  featnames()
t_feats 
# non-terrorism data features:
nt_feats <- dfm(nonter_toks, verbose = TRUE) %>%
  dfm_trim(min_termfreq = 5) %>%
  featnames()
nt_feats
# full df feats:
df_feats <- dfm(df_toks, verbose = TRUE) %>%
  dfm_trim(min_termfreq = 5) %>%
  featnames()
df_feats

# Leave the pads so that non-adjacent words will not become adjacent #######################################################
# (Replaces empty strings if tokens are removed later so distances aren't affected) 
ter_toks <- tokens_select(ter_toks, t_feats, padding = TRUE)
ter_toks

nonter_toks <- tokens_select(nonter_toks, nt_feats, padding = TRUE)
nonter_toks

df_toks <- tokens_select(df_toks, t_feats, padding=TRUE)
df_toks

# Construct the feature co-occurrence matrix ###############################################################################
# FCM gives counts of how often words appear together in the text

# terrorism data FCM:
ter_fcm <- fcm(ter_toks, 
               context = "window", # window for co-occurrence within a defined window of words
               count = "weighted", # weighted as a function of distance between words
               weights = 1 / (1:5), tri = TRUE) # window of 5, weights decrease as distance grows
ter_fcm

# non-terrorism data FCM:
nonter_fcm <- fcm(nonter_toks, context = "window", count = "weighted", weights = 1 / (1:5), tri = TRUE)
nonter_fcm

# full data FCM:
df_fcm <- fcm(df_toks, context="window", count="weighted", weights=1/(1:5), tri=TRUE)
df_fcm

# It's time to fit the word embedding GloVe models ###########################################################
# GloVe model learns how words co-occur together and uses those representations to demonstrate how words are related

# terrorism data GLOVE:
ter_glove <- GlobalVectors$new(rank = 50, x_max = 10)
ter_wv_main <- ter_glove$fit_transform(ter_fcm, n_iter = 10,
                                       convergence_tol = 0.01, n_threads = 8) # INFO  [19:22:37.469] epoch 10, loss 0.0179 

dim(ter_wv_main) # [1] 861  50
ter_wv_main

# non-terrorism GLOVE:
nonter_glove <- GlobalVectors$new(rank = 50, x_max = 10)
nonter_wv_main <- nonter_glove$fit_transform(nonter_fcm, n_iter = 10,
                                             convergence_tol = 0.01, n_threads = 8) # INFO  [19:23:13.281] epoch 10, loss 0.0171 

dim(nonter_wv_main) # [1] 995  50
nonter_wv_main

# full df GLOVE:
full_glove <- GlobalVectors$new(rank = 50, x_max = 10)
full_wv_main <- full_glove$fit_transform(nonter_fcm, n_iter = 10,
                                             convergence_tol = 0.01, n_threads = 8) # INFO  [19:23:36.599] epoch 10, loss 0.0174

dim(full_wv_main) # [1] 995  50
full_wv_main

# Averaging learned word vectors: ###########################################################################################
#   The two vectors are main and context. 
#   Main: rows=words, context: columns=words
#   Averaging the two word vectors results in more accurate representation

# terrorism averaged learned word vectors:
ter_wv_context <- ter_glove$components
dim(ter_wv_context) # 50 861
ter_word_vectors <- ter_wv_main + t(ter_wv_context)
ter_word_vectors

# non-terrorism averaged learned word vectors:
nonter_wv_context <- nonter_glove$components
dim(nonter_wv_context) # 50 995
nonter_word_vectors <- nonter_wv_main + t(nonter_wv_context)
nonter_word_vectors

# full df averaged learned word vectors:
full_wv_context <- full_glove$components
dim(full_wv_context) # 50 995
full_word_vectors <- full_wv_main + t(full_wv_context)
full_word_vectors


# Examining term representations  ###########################################################################################
# Here, we're finding word vectors that are the most similar to each other

# (To see words included in each vector:
# rownames(ter_word_vectors), rownames(nonter_word_vectors), rownames(full_word_vectors)

# Below, I inspect how words are used differently by members of terrorist organizations and non-terrorist organizations
# We inspect the Euclidean distance of word embeddings that are common in both groups 
# Words = NEED, LEAD, ATTACK, POWER, BELIEVE

# EUCLIDEAN DISTANCE: Terrorism data vs Non-terrorism data, word="NEED" ##########################################
dist_full_need_ter <- as.matrix(dist(as.dfm(ter_word_vectors)),upper=T)
head(sort(dist_full_need_ter["need",], decreasing = F), 10) 
# need      prevent   despite      also    Forces   form      Treatment      Said     start   without 
# 0.000000  2.892823  3.157322  3.236255  3.251301  3.264365  3.265608  3.284359  3.294062  3.308139 

dist_full_need_nonter <- as.matrix(dist(as.dfm(nonter_word_vectors)),upper=T)
head(sort(dist_full_need_nonter["need",], decreasing = F), 10)
# need    liberation   standing      Force    Ecological      front       make      bread        set        say 
# 0.000000   2.877761   3.028126   3.048663   3.102208   3.122635   3.162593   3.166958   3.189529   3.199402 

# EUCLIDEAN DISTANCE: Terrorism data vs Non-terrorism data, word="LEAD" ########################################## *
dist_full_lead_ter <- as.matrix(dist(as.dfm(ter_word_vectors)),upper=T)
head(sort(dist_full_lead_ter["lead",], decreasing = F), 10) 
#      lead      stop       act      Jews    People         o     using    effort     Black   situation 
#   0.000000  2.985320  3.038663  3.053134  3.143034  3.151942  3.165137  3.173750  3.222717  3.237177 

dist_full_lead_nonter <- as.matrix(dist(as.dfm(nonter_word_vectors)),upper=T)
head(sort(dist_full_lead_nonter["lead",], decreasing = F), 10)
#      lead       ready       point    humanity      wanted        join cooperation      defend       right         Law 
#   0.000000    2.930431    2.933512    2.959091    2.979902    2.998487    3.052167    3.104197    3.111599    3.119664 

# EUCLIDEAN DISTANCE: Terrorism data vs Non-terrorism data, word="ATTACK" ########################################## *
dist_full_attack_ter <- as.matrix(dist(as.dfm(ter_word_vectors)),upper=T)
head(sort(dist_full_attack_ter["attack",], decreasing = F), 10) 
#      lead      stop       act      Jews    People         o     using    effort     Black situation 
#   0.000000  2.985320  3.038663  3.053134  3.143034  3.151942  3.165137  3.173750  3.222717  3.237177 

dist_full_attack_nonter <- as.matrix(dist(as.dfm(nonter_word_vectors)),upper=T)
head(sort(dist_full_attack_nonter["attacks",], decreasing = F), 10)
#      lead       ready       point    humanity      wanted        join cooperation      defend       right         Law 
#   0.000000    2.930431    2.933512    2.959091    2.979902    2.998487    3.052167    3.104197    3.111599    3.119664 

# EUCLIDEAN DISTANCE: Terrorism data vs Non-terrorism data, word="POWER" ##########################################
dist_ter_power <- as.matrix(dist(as.dfm(ter_word_vectors)),upper=T)
head(sort(dist_ter_power["power",], decreasing = F), 10)
#  power    source    animal     alone      Born   visited      Give      Even activists    others 
# 0.000000  2.995399  3.239560  3.254656  3.313893  3.316896  3.354805  3.390962  3.394417  3.396537 

dist_nonter_power <- as.matrix(dist(as.dfm(nonter_word_vectors)),upper=T)
head(sort(dist_nonter_power["power",], decreasing = F), 10)
#     power          TV        vans         end        shut      ISKCON brotherhood         lot        Also        Help 
# 0.000000    2.856714    2.914575    2.933928    2.941528    2.970983    2.972782    2.978960    3.004142    3.081994 

# EUCLIDEAN DISTANCE: Terrorism data vs Non-terrorism data, word="BELIEVE" ##########################################
dist_ter_believe <- as.matrix(dist(as.dfm(ter_word_vectors)),upper=T)
head(sort(dist_ter_believe["believe",], decreasing = F), 10)
#  believe    using  message     tell   opened      run      say     Stay   humans    night 
# 0.000000 3.117470 3.406645 3.423482 3.458611 3.462165 3.464435 3.483720 3.504133 3.511006 

dist_nonter_believe <- as.matrix(dist(as.dfm(nonter_word_vectors)),upper=T)
head(sort(dist_nonter_believe["believe",], decreasing = F), 10)
#   believe   Therefore       along Non-violent        shut        free         air      values           5      behind 
# 0.000000    2.892120    3.027276    3.067086    3.153812    3.158834    3.173746    3.186372    3.189476    3.193404 


#terrorism 50-word plot: ###########################
#grab 50 words
ter_forplot<-as.data.frame(ter_wv_main[50:100,])
ter_forplot$word<-rownames(ter_forplot)

#now plot
x2 <- ggplot(ter_forplot, aes(x=V1, y=V2, label=word))+
  geom_text(aes(label=word),hjust=0, vjust=0, color="blue")+
  theme_minimal()+
  xlab("First Dimension Created by GloVe")+
  ylab("Second Dimension Created by GloVe")


# non-terrorism 50-word plot: ###########################
#grab 50 words
nonter_forplot<-as.data.frame(nonter_wv_main[50:100,])
nonter_forplot$word<-rownames(nonter_forplot)

#now plot
y2 <- ggplot(nonter_forplot, aes(x=V1, y=V2, label=word))+
  geom_text(aes(label=word),hjust=0, vjust=0, color="blue")+
  theme_minimal()+
  xlab("First Dimension Created by GloVe")+
  ylab("Second Dimension Created by GloVe")

# full df 50-word plot: ###########################
#grab 50 words
full_forplot<-as.data.frame(full_wv_main[50:100,])
full_forplot$word<-rownames(full_forplot)

#now plot
ggplot(full_forplot, aes(x=V1, y=V2, label=word))+
  geom_text(aes(label=word),hjust=0, vjust=0, color="blue")+
  theme_minimal()+
  xlab("First Dimension Created by GloVe")+
  ylab("Second Dimension Created by GloVe")

### TO-DO:
# change colors of words based on document Type in plot above

# Predictive analysis below #######################################################################################
#     Combined the ter and non-ter corpora, 
#     Fit a single glove embedding to the complete corpus, 
#     Averaged the vectors for words in each document, 
#     Predicted using naive Bayes whether or not a document is a terrorism document based on the embedding positions


# fit single glove embedding to complete corpus:
full_fcm <- fcm(df_toks, context = "window", count = "weighted", weights = 1 / (1:5), tri = TRUE)
full_fcm # feature co-occurence matrix for the entire dataset

glove <- GlobalVectors$new(rank = 50, x_max = 10)
glove_wv_main <- glove$fit_transform(full_fcm, n_iter=10,
                                     convergence_tol=0.01, n_threads=8) # INFO  [10:40:52.359] epoch 10, loss 0.0218 

# Average vectors for words:
wv_context <- glove$components
dim(wv_context) # [1]  50 938
word_vectors <- glove_wv_main + t(wv_context)

# split data 70 training/30 testing
smp_size <- floor(0.7 * nrow(glove_wv_main))
set.seed(1234) # set the seed for reproducibility
train_ind <- sample(seq_len(nrow(glove_wv_main)), size = smp_size)

train70 <- fulldf[train_ind, ] # 70% training df 
test30 <- fulldf[-train_ind, ] # 30% testing df 


# Naive Bayes model, fit to training data:
type_rec <- 
  recipe(Type ~ rawtext, data=train70)

#install.packages("textrecipes")
library(textrecipes)
#install.packages("naivebayes")
library(naivebayes)
#install.packages("discrim")
library(discrim)

type_rec <- type_rec %>%
  step_tokenize(rawtext) %>%
  step_tokenfilter(rawtext, max_tokens = 1e3) %>%
  step_tfidf(rawtext)

#type_rec <- type_rec %>%
# step_tfidf()

type_wf <- workflow() %>%
  add_recipe(type_rec)

nb_spec <- naive_Bayes() %>%
  set_mode("classification") %>%
  set_engine("naivebayes")
nb_spec

nb_fit <- type_wf %>%
  add_model(nb_spec) %>%
  fit(data = train70)
# We have trained the classification model!


# Now evaluate the model's performance on the test set: 
set.seed(234)

type_folds <- vfold_cv(train70)
type_folds # 10-fold cross-validation

nb_wf <- workflow() %>%
  add_recipe(type_rec) %>%
  add_model(nb_spec)
nb_wf

nb_rs <- fit_resamples(
  nb_wf,
  type_folds,
  control = control_resamples(save_pred = TRUE)
)

nb_rs_metrics <- collect_metrics(nb_rs)
nb_rs_predictions <- collect_predictions(nb_rs)

nb_rs_metrics
# ROC AUC gives us the performance parameters for binary classification on the test set
# average accuracy of 54% -- not great

# let's visualize the ROC curve:
nb_rs_predictions %>%
  group_by(id) %>%
  roc_curve(truth = Type, .pred_Terrorist) %>%
  autoplot() +
  labs(
    color = NULL,
    title = "ROC curve for Terrorism Documents",
    subtitle = "Each resample fold is shown in a different color"
  )
# we see that the "curves" (not super curve-like here, but still) are close to the diagonal line, which means the model's predictions aren't much better than just random guessing at this point 

# Sensitivity: the ability of the model to correctly identify text as Terrorist 
# Specificity: the ability of the model to correctly identify text as Non-Terrorist

# Each fold is SUPER different
# Next steps to improve the predictive performance of the model =
# 1) collect more data (small sample size now, n=151 documents)
# 2) add weights to specific words
# 3) remove words from analyses that weren't removed earlier with stopwords function (see list at 468)



# Thank you! ##################################################################################################
############################################################################################################### 
############################################################################################################### 
############################################################################################################### 
############################################################################################################### 











### TO-DO:
# add weights to motives-related words from LIWC motives dictionaries
# to make the case that a) we can learn about personality from text and b) we can use motives from text to understand psyche of terrorists

### TO-DO: REMOVE THESE WORDS FROM TEXT:
# about
# after
# also
# and
# any
# are
# because
# been
# but 
# can
# for
# from
# get
# had
# has
# have
# how
# into
# its
# said
# say
# than
# that
# the
# was
# which
# would


# To overlay scatterplots in R

# import the required libraries
#library(ggplot2)
#library(reshape2)

# assign data
#a1=ter_wv_main
#a2=nonter_wv_main

# create a dataframe from combined data
# and set count to however many points are in each dataset
#plotdf = data.frame(a1, a2, count = c(1:2))
#names(plotdf)

# melt the dataframe
#df.m = melt(plotdf, id.vars ="count", measure.vars = c("V1","V2"))

# take a look at what melt() does to get an idea of what is going on
#df.m

# plot out the melted dataframe using ggplot
#ggplot(df.m, aes(count, value, colour = variable)) + geom_point() + ylim(-2,2)

# swapping the axis
#ggplot(df.m, aes(value, count, colour = variable)) + geom_point() + xlim(-2,2)


