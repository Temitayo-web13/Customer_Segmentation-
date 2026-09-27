library(readxl)
library(dplyr)
library(readr)
library(ggplot2) 
library(lubridate)
library(factoextra)
library(cluster) 
library(gridExtra) 


data <-read_excel("Online Retail.xlsx")
head(data)
tail(data)
dim(data)
names(data)
str(data) 
colSums(is.na(data))

#cleaning the data
data <- data %>%
  filter(!is.na(CustomerID))
data <- data %>%
  filter(!grepl("^C", InvoiceNo))
data <- data %>%
  filter(Quantity>0)
data <- data %>% 
  filter(UnitPrice>0)
data

colSums(is.na(data))


#calculating the revenue
data <- data %>%
  mutate(Revenue=Quantity * UnitPrice)
nrow(data)
table(data$Country)
summary(data$Quantity)
dim(data)
head(data)
sum(data$Revenue)
mean(data$Revenue)

#Exploring the dataset
n_distinct(data$CustomerID)
nrow(data)
table(data$Country)
customer_revenue <- data %>%
  group_by(CustomerID) %>%
  summarise(TotalRevenue = sum(Revenue)) %>%
  arrange(desc(TotalRevenue))
head(customer_revenue, 10)
n_distinct(data$CustomerID)

##creating rfm features
reference_date <- max(data$InvoiceDate) + days(1)
rfm <- data %>%
  group_by(CustomerID) %>%
  summarise( 
    Recency=as.numeric(difftime(
      reference_date,
      max(InvoiceDate),
      units="days"
    )),
    Frequency =n_distinct (InvoiceNo),
    Monetary=sum(Revenue)
  )
head(rfm)


#visualizing the rfm variables
ggplot(rfm, aes(x = Recency)) +
  geom_histogram(bins =30)

ggplot(rfm, aes(x = Frequency)) + 
  geom_histogram(bins = 30)

ggplot(rfm, aes(x = Monetary)) + 
  geom_histogram(bins=30)

#selecting variable for clustering
rfm_features <- rfm %>% 
  select(Recency,Frequency,Monetary)

#scaling the data
rfm_scaled <- scale(rfm_features) 

#choosing the number of clusters(k)
elbow_plot<-fviz_nbclust (
  rfm_scaled,
  kmeans,
  method ="wss",
  k.max=10
)

silhouette_plot<- fviz_nbclust(
  rfm_scaled,
  kmeans,
  method ="silhouette",
  k.max=10
)

grid.arrange(elbow_plot, silhouette_plot, nrow = 2)


#run k means
set.seed(123) 
kmeans_model <- kmeans( 
  rfm_scaled, 
  centers = 3, 
  nstart = 25, 
  iter.max = 100 
) 
#see which customer belong to which cluster
rfm$Cluster <- kmeans_model$cluster 
table(rfm$Cluster) 
head(rfm)

#Examine the cluster centres
kmeans_model$centers 

#visualize the clusters
fviz_cluster( 
  kmeans_model,   
  data = rfm_scaled,
  geom = "point" 
) 

#Evaluating the cluster
table(rfm$Cluster)
cluster_profile <- rfm %>%   
  group_by(Cluster) %>%  
  summarise(    
    Customers = n(),
    AvgRecency = mean(Recency),     
    AvgFrequency = mean(Frequency),   
    AvgMonetary = mean(Monetary)   
  )  
cluster_profile

#silhouette score
sil <- silhouette(   
  kmeans_model$cluster,   
  dist(rfm_scaled) 
)  
mean(sil[, "sil_width"])  
fviz_silhouette(sil)

#creating meaningful segment names
rfm <- rfm %>%  
  mutate(    
    Segment = case_when(      
      Cluster == 1 ~ "Dormant / Low-Value Customers",       
      Cluster == 2 ~ "Core / Occasional Customers",       
      Cluster == 3 ~ "VIP / High-Value Loyal Customers" 
    ) 
  ) 

#summarizing the segments
segment_summary<- rfm %>%
  count(Segment) %>%
  mutate(Percentage=n/sum(n) *100)

segment_summary
final_profile<- rfm %>%
  group_by(Segment)  %>%
  summarise(
    Customers =n(),
    AvgRecency =round(mean(Recency), 1),
    AvgFrequency =round(mean(Frequency), 1),
    AvgMonetary =round(mean(Monetary), 2),
    TotalRevenue=round(sum(Monetary),2)
  )
final_profile

#extracting the cleaned data
library(writexl)
write_xlsx(data, "Cleaned_Online_Retail.xlsx")




    
    
