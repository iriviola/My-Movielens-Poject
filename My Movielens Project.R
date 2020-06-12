####################################
#Capstone Movielens Project
#Irene Viola
#12 june 2020
#########QUESTION##################
#Use Machine learning algorithms on a subset of Movielens to predict the ratings and calculate RMSE

##############################################################
# load data sets (edx, validation) and libraries
# Loads the data set used in the Movielens assessment
################################
# Create edx training set, validation test set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
######################################################################################################
#############
#Check Average method#
#############
# average rating mu #
mu <- mean(edx$rating)
#Predict unkown rating and calculate RMSE#
#############
#Consider Movie effect#
#############
# Calculate the movie effect, b_m
b_m <- edx %>%
       group_by(movieId) %>%
       summarize(b_m = mean(rating - mu),.groups ='drop')
#Now predict unknow rating using mu and b_m
p_ratings_M <- validation %>% 
             left_join(b_m, by='movieId') %>%
             mutate(pred = mu + b_m) %>%
             pull(pred)
#Then calculate the RMSE for the p_rating_M
RMSE_M<- RMSE(validation$rating, p_ratings_M)
#Plot b_m 
plot_b_m <- qplot(b_m, data = b_m, bins = 15, color = I("black"))
###############################
# Now consider the movie + user effect
###############################
#First calculate the user effect, b_u
b_u <- edx %>% 
       left_join(b_m, by='movieId') %>%
       group_by(userId) %>%
       summarize(b_u = mean(rating - mu - b_m),.groups ='drop')
#Make a new prediction considering the user effect and the movie effect
p_ratings_M_U <- validation %>% 
                left_join(b_m, by='movieId') %>%
                left_join(b_u, by='userId') %>%
                mutate(pred = mu + b_m + b_u) %>%
                pull(pred)
#Then calculate RMSE for user and movie effect ratings prediction
RMSE_M_U<- RMSE(validation$rating, p_ratings_M_U)
#plot b_u
plot_b_u <- qplot(b_u, data = b_u, bins = 15, color = I("black"))
#######################
# Regularize movie + user effect 
#######################
# Need to use lambda and need to find out which is the best value
#first define possible lambdas sequence
lambdas <- seq(from=0, to=10, by=0.25)
#Define RMSE function on user + movie effect and repeat for each lambdas of the sequence 
rmses <- sapply(lambdas, function(l){
          mu <- mean(edx$rating) # average rating across training data
          b_m <- edx %>% #Movie effect 
          group_by(movieId) %>%
          summarize(b_m = sum(rating - mu)/(n()+l), .groups ='drop')
          b_u <- edx %>% #User effect
          left_join(b_m, by="movieId") %>%
          group_by(userId) %>%
          summarize(b_u = sum(rating - b_m - mu)/(n()+l), .groups ='drop')
          #Prediction user+movie effect
          pr_ratings <- validation %>% 
          left_join(b_m, by = "movieId") %>%
          left_join(b_u, by = "userId") %>%
          mutate(pred = mu + b_m + b_u) %>%
          pull(pred)
          #RMSE for each prediction each lambda movie+ user effect on validation set
          return(RMSE(validation$rating, pr_ratings)) 
          })
#To define the best lambda, plot RMSE vs lambdas
qplot(lambdas, rmses)
# and print minimum RMSE that will give you the best lambda to use for regularization
min(rmses)
#########################
# Model with regularised movie + user effects 
########################
#best lambda
lambda <- lambdas[which.min(rmses)]
#Movie effect regularised
b_m <- edx %>% 
       group_by(movieId) %>%
       summarize(b_m = sum(rating - mu)/(n()+lambda), .groups ='drop')
#User effect regularised
b_u <- edx %>% 
       left_join(b_m, by="movieId") %>%
       group_by(userId) %>%
       summarize(b_u = sum(rating - b_m - mu)/(n()+lambda), .groups ='drop')
#Rating prediction on validation set using regularised terms
pr_ratings_U_M2 <- validation %>% 
                     left_join(b_m, by = "movieId") %>%
                     left_join(b_u, by = "userId") %>%
                     mutate(pred = mu + b_m + b_u) %>%
                     pull(pred)
# RMSE predictions regularised
RMSE_U_M2 <- RMSE(validation$rating, pr_ratings_U_M2)
###############################
# Now consider the movie + user + genre effect
###############################
#First calculate the genre effect b_g
b_g <- edx %>% # Genre effect
  left_join(b_m, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_m - b_u), .groups ='drop')
#plot b_g
plot_g <- qplot(b_g, data = b_g, bins = 15, color = I("black"))

###############################
# Now regularize the movie + user + genre effect to find correct lambda
###############################
#first define a new sequence of lambdas 
lambdas2 <- seq(0, 20, 1)
#Define a new RMSE function on user + movie+ genre effect and repeat for each lambdas
rmses2 <- sapply(lambdas2, function(l){
         mu <- mean(edx$rating)# average rating across training data
         b_m <- edx %>% #Movie effect 
         group_by(movieId) %>%
         summarize(b_m = sum(rating - mu)/(n()+l), .groups ='drop')
         b_u <- edx %>% #User effect
         left_join(b_m, by="movieId") %>%
         group_by(userId) %>%
         summarize(b_u = sum(rating - b_m - mu)/(n()+l), .groups ='drop')
         b_g <- edx %>% # Genre effect
         left_join(b_m, by='movieId') %>%
         left_join(b_u, by='userId') %>%
         group_by(genres) %>%
         summarize(b_g = sum(rating - mu - b_m - b_u)/(n()+l), .groups ='drop')
         #Prediction movie+user+genre effect
         pr_ratings_U_M_G <- validation %>% 
         left_join(b_m, by='movieId') %>%
         left_join(b_u, by='userId') %>%
         left_join(b_g, by = 'genres') %>%
         mutate(pred = mu + b_m + b_u + b_g) %>% 
         pull(pred)
          
        return(RMSE(validation$rating,pr_ratings_U_M_G))
           })
#To define the best new lambda, plot RMSE vs lambdas
qplot(lambdas2, rmses2)
# and print minimum RMSE that will give me the best new lambda 
min(rmses2)
#########################
# Model with regularised movie + user + genre effects 
########################
#best lambda
lambda2 <- lambdas[which.min(rmses2)]
#Regularised movie effect using lambda2
b_m2 <- edx %>%
        group_by(movieId) %>% 
        summarize(b_m = sum(rating - mu)/(n()+lambda2), n_i = n(), .groups ='drop')
#Regularised user effect using lambda2
b_u2 <- edx %>% 
        left_join(b_m2, by='movieId') %>%
        group_by(userId) %>%
        summarize(b_u = sum(rating - mu - b_m)/(n()+lambda2), n_u = n(), .groups ='drop')
#Regularised genre effect using lambda2
b_g2 <- edx %>%
        left_join(b_m2, by='movieId') %>%
        left_join(b_u2, by='userId') %>%
        group_by(genres) %>%
        summarize(b_g = sum(rating - mu - b_m - b_u)/(n()+lambda2), n_g = n(), .groups ='drop')
#Rating prediction on validation set using regularised terms with lambda2
pr_ratings_U_M_G2 <- validation %>% 
                  left_join(b_m2, by='movieId') %>%
                  left_join(b_u2, by='userId') %>%
                  left_join(b_g2, by = 'genres') %>%
                  mutate(pred = mu + b_m + b_u + b_g) %>% 
                  pull(pred)
# RMSE predictions FINAL
RMSE_U_M_G2 <- RMSE(validation$rating,pr_ratings_U_M_G2)


