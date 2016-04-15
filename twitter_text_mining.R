# Most of the code came from this fantastic text mining tutorial: 
# http://www2.rdatamining.com/uploads/5/7/1/3/57136767/rdatamining-slides-text-mining.pdf
# courtesy of Yanchang Zhao

library("ggplot2")
library("dplyr")
library("plyr")
library("ggmap")
library("tm")
library("igraph")
setwd("LOCATION_OF_TWEETS")

# Read in the tweets
df <- read.csv("TWEETS",header=F)
names(df) <- c("Classification","Confidence","TweetID","Tweet","Account","DateTime","UNIXtime","Latitude","Longitude")
df$UNIXtime <- df$UNIXtime/1000
df$UNIXtime <- as.POSIXct(df$UNIXtime, origin = "1970-01-01") # Convert from UNIX time

# Study terms in tweets
myCorpus <- Corpus(VectorSource(df$Tweet)) # Load all tweets to corpus
myCorpus <- tm_map(myCorpus, content_transformer(tolower)) # Make lower case
# Remove URLs
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
myCorpus <- tm_map(myCorpus, content_transformer(removeURL))
# Remove anything other than English letters or space
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
myCorpus <- tm_map(myCorpus, content_transformer(removeNumPunct))
myStopwords <- stopwords("english")
myCorpus <- tm_map(myCorpus, removeWords, myStopwords) # Remove Stop words
myCorpus <- tm_map(myCorpus, stripWhitespace) # Remove extra white space

# Stem words
myCorpus <- tm_map(myCorpus, stemDocument)
stemCompletion2 <- function(x, dictionary) {
  x <- unlist(strsplit(as.character(x), " "))
  x <- x[x != ""]
  x <- stemCompletion(x, dictionary=dictionary)
  x <- paste(x, sep="", collapse=" ")
  PlainTextDocument(stripWhitespace(x))
}

# Create Term Document Matrix
tdm <- TermDocumentMatrix(myCorpus,control = list(wordLengths = c(1, Inf)))

# Look at most frequent terms
freq.terms <- findFreqTerms(tdm, lowfreq = 15000) # User-defined lowfreq
term.freq <- rollup(tdm,2,na.rm=TRUE,fun=sum)
term.freq <- as.matrix(term.freq)
term.freq <- subset(term.freq, term.freq >= 15000)
mostFreq <- data.frame(term = rownames(term.freq), freq = term.freq)
mostFreq$X1<-mostFreq$X1/1000

# Plot most frequent terms in TDM
ggplot(mostFreq, aes(x = term, y = X1)) +
  geom_bar(stat = "identity") + xlab("Terms") + ylab("Count (thousands)") + coord_flip()

# Associations
findAssocs(tdm, "london", 0.25)

# Plot network
freqtdm <- inspect(tdm[freq.terms,]) # Make TDM of just most frequent terms
termMatrix <- freqtdm %*% t(freqtdm)
g <- graph.adjacency(termMatrix, weighted=T, mode = "undirected")
g <- simplify(g)
# Set labels and degrees of vertices
V(g)$label <- V(g)$name
V(g)$degree <- degree(g)
V(g)$label.cex <- 2.3*V(g)$degree/max(V(g)$degree+.3)
V(g)$label.color <- rgb(0, 0, .2, .8)
V(g)$frame.color <- NA
# Plot graph
set.seed(3952)
layout1 <- layout.fruchterman.reingold(g)
plot(g, layout=layout1)

# Create dendrogram
tdm2 <- removeSparseTerms(tdm, sparse = 0.98)
m2 <- as.matrix(tdm2)
distMatrix <- dist(scale(m2))
fit <- hclust(distMatrix, method = "ward.D")
plot(fit)
rect.hclust(fit,k=6)

#########################################################################

# Assign pos/neg tweet counts to each day in the data
days <- unique(substr(df$UNIXtime,1,10))[c(1:5,7:109)]
pos <- vector(length=length(days))
neg <- vector(length=length(days))
vol <- vector(length=length(days))
for(j in 1:nrow(df)){
  print(j)
  if(toString(df$UNIXtime[j])=="NA"){
    print("NA")
  }else{
    for(i in 1:length(days)){
      if(substr(as.character(df$UNIXtime[j]),1,10) == days[i]){
        vol[i] = vol[i] + 1
        if(df$Classification[j] == "pos"){
          pos[i] = pos[i] + 1
        }else{
          neg[i] = neg[i] + 1
        }
      }else{
        print(substr(df$UNIXtime[j],1,10))
      }
    }
  }
}


# Plot relative sentiment over time
qplot(as.Date(days),pos/(pos+neg))
qplot(as.Date(days),vol)


## Plot grid on London
grid<-read.csv("LOCATION_OF_GRID_COORDS",sep="\t",header=F)
r1 <- c(max(grid$V1), min(grid$V2))
r2 <- c(min(grid$V1), max(grid$V2))
grid <- rbind(grid,r1,r2)
line <- rbind(filter(grid,V1==max(V1)),filter(grid,V1==min(V1)),filter(grid,V2==max(V2)))
map <- get_map(location = c(lon=mean(grid$V2),lat=mean(grid$V1)), zoom = 9)
ggmap(map)+geom_point(aes(x=V2,y=V1),data=grid,size=1)
