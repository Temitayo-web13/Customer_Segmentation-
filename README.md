# E-commerce Customer Segmentation Using K-means Clustering

## Project Overview

This project uses **R** to analyse e-commerce customer purchasing
behaviour and divide customers into meaningful groups using **K-means
clustering**.

The project follows an **RFM (Recency, Frequency, Monetary)** approach:

-   **Recency** --- how recently the customer purchased.
-   **Frequency** --- how often the customer purchased, measured using
    unique invoice numbers.
-   **Monetary** --- how much the customer spent.

The main business question is:

> Can customer purchasing behaviour be used to divide customers into
> useful groups that can support marketing decisions?

Because the dataset has no predefined customer segments, this is an
**unsupervised machine learning** problem.

------------------------------------------------------------------------

## Business Problem

An online retailer can have thousands of transactions without knowing
which customers are highly valuable, loyal, occasional, or becoming
inactive.

This project uses transaction data to discover customer groups with
similar purchasing behaviour and translate those groups into practical
marketing actions.

The complete workflow is:

**Transaction data → Cleaning → Revenue → RFM → Scaling → K-means →
Evaluation → Segments → Business recommendations**

------------------------------------------------------------------------

## Dataset

The project uses the **Online Retail transaction dataset**. Each row
represents a transaction line rather than a complete customer.

Important variables include:

  Variable        Description
  --------------- ------------------------------
  `InvoiceNo`     Invoice/order number
  `StockCode`     Product code
  `Description`   Product description
  `Quantity`      Number of items purchased
  `InvoiceDate`   Date and time of transaction
  `UnitPrice`     Price per item
  `CustomerID`    Customer identifier
  `Country`       Customer\'s country

------------------------------------------------------------------------

## Tools and Packages

The analysis was developed in **R** using:

-   `readxl`
-   `dplyr`
-   `ggplot2`
-   `lubridate`
-   `factoextra`
-   `cluster`
-   `gridExtra`

Methods used:

-   Data cleaning
-   Exploratory data analysis
-   Revenue calculation
-   RFM analysis
-   Feature scaling
-   Elbow analysis
-   Silhouette analysis
-   K-means clustering
-   Cluster profiling
-   Business interpretation

------------------------------------------------------------------------

# 1. Loading and Inspecting the Data {#1-loading-and-inspecting-the-data}

The Excel dataset is loaded with:

``` r
data <- read_excel("Online Retail.xlsx")
```

The project uses `head()`, `tail()`, `dim()`, `names()`, `str()` and
`colSums(is.na())` to understand the data structure, dimensions,
variables and missing values.

------------------------------------------------------------------------

# 2. Data Cleaning {#2-data-cleaning}

The following filters were applied:

``` r
data <- data %>%
  filter(!is.na(CustomerID))

data <- data %>%
  filter(!grepl("^C", InvoiceNo))

data <- data %>%
  filter(Quantity > 0)

data <- data %>%
  filter(UnitPrice > 0)
```

### Cleaning decisions

-   Rows without `CustomerID` were removed because customers must be
    identifiable for RFM analysis.
-   Invoices beginning with `C` were treated as cancellations and
    removed.
-   Zero or negative quantities were removed.
-   Zero or negative prices were removed.

This produces a dataset focused on valid positive customer purchases.

------------------------------------------------------------------------

# 3. Calculating Revenue {#3-calculating-revenue}

Revenue was calculated as:

``` r
data <- data %>%
  mutate(Revenue = Quantity * UnitPrice)
```

Therefore:

**Revenue = Quantity × UnitPrice**

Revenue is later summed at customer level to form the Monetary RFM
feature.

The cleaned data was also explored using country counts, quantity
summaries, total revenue and average transaction-line revenue.

The country output shows that the **United Kingdom has the largest
number of transaction records, with 354,321 records**.

The Quantity summary was:

  Statistic         Value
  -------------- --------
  Minimum               1
  1st Quartile          2
  Median                6
  Mean              12.99
  3rd Quartile         12
  Maximum          80,995

The large maximum relative to the median shows that transaction
quantities have a strong upper tail.

------------------------------------------------------------------------

# 4. Customer Exploration {#4-customer-exploration}

Customer-level revenue was calculated with:

``` r
customer_revenue <- data %>%
  group_by(CustomerID) %>%
  summarise(TotalRevenue = sum(Revenue)) %>%
  arrange(desc(TotalRevenue))
```

This makes it possible to examine the highest-revenue customers before
performing formal clustering.

------------------------------------------------------------------------

# 5. Creating RFM Features {#5-creating-rfm-features}

The transaction data is converted into one row per customer.

### Recency

A reference date is defined as one day after the latest transaction:

``` r
reference_date <- max(data$InvoiceDate) + days(1)
```

Lower Recency means a more recent purchase.

### Frequency

Frequency is the number of unique invoices:

``` r
Frequency = n_distinct(InvoiceNo)
```

### Monetary

Monetary is total customer revenue:

``` r
Monetary = sum(Revenue)
```

The full calculation is:

``` r
rfm <- data %>%
  group_by(CustomerID) %>%
  summarise(
    Recency = as.numeric(difftime(
      reference_date,
      max(InvoiceDate),
      units = "days"
    )),
    Frequency = n_distinct(InvoiceNo),
    Monetary = sum(Revenue)
  )
```

------------------------------------------------------------------------

# 6. Exploring the RFM Variables {#6-exploring-the-rfm-variables}

Histograms were created for Recency, Frequency and Monetary.

``` r
ggplot(rfm, aes(x = Recency)) +
  geom_histogram(bins = 30)

ggplot(rfm, aes(x = Frequency)) +
  geom_histogram(bins = 30)

ggplot(rfm, aes(x = Monetary)) +
  geom_histogram(bins = 30)
```

### Recency

The distribution covers a wide range of inactivity periods. This shows
substantial variation in how recently customers purchased.

### Frequency

Frequency is strongly concentrated at lower values, with a smaller group
of customers making many more purchases.

### Monetary

Monetary is strongly right-skewed. Most customers have relatively lower
spending, while a small number have very high spending.

These distributions provide a useful reason for segmenting customers
based on RFM behaviour.

------------------------------------------------------------------------

# 7. Selecting and Scaling the RFM Variables {#7-selecting-and-scaling-the-rfm-variables}

The clustering features are:

``` r
rfm_features <- rfm %>%
  select(Recency, Frequency, Monetary)
```

They are then scaled:

``` r
rfm_scaled <- scale(rfm_features)
```

Scaling is important because K-means uses distance. Monetary has a much
larger numerical range than Frequency, so using raw values could allow
Monetary to dominate the distance calculation.

------------------------------------------------------------------------

# 8. Choosing the Number of Clusters {#8-choosing-the-number-of-clusters}

The project investigates K using both the **Elbow method** and
**Silhouette method**.

``` r
elbow_plot <- fviz_nbclust(
  rfm_scaled,
  kmeans,
  method = "wss",
  k.max = 10
)

silhouette_plot <- fviz_nbclust(
  rfm_scaled,
  kmeans,
  method = "silhouette",
  k.max = 10
)
```

The final model was run with **K = 3**.

------------------------------------------------------------------------

# 9. Running K-means {#9-running-k-means}

``` r
set.seed(123)

kmeans_model <- kmeans(
  rfm_scaled,
  centers = 3,
  nstart = 25,
  iter.max = 100
)
```

-   `centers = 3`: three clusters.
-   `nstart = 25`: 25 starting configurations.
-   `iter.max = 100`: up to 100 adjustment iterations per start.
-   `set.seed(123)`: reproducible results.

Cluster assignments are stored using:

``` r
rfm$Cluster <- kmeans_model$cluster
```

The resulting cluster sizes are:

-   Cluster 1: **1,087 customers**
-   Cluster 2: **3,225 customers**
-   Cluster 3: **26 customers**

------------------------------------------------------------------------

# 10. Cluster Centres {#10-cluster-centres}

The scaled cluster centres are:

  Cluster     Recency   Frequency   Monetary
  --------- --------- ----------- ----------
  1            -1.540      -0.350     -0.158
  2            -0.512       0.053     -0.022
  3            -0.866       8.074      9.328

These are standardised values rather than raw RFM values.

Cluster 3 is particularly distinctive because of its very high scaled
Frequency and Monetary values. This supports its interpretation as the
VIP/high-value group.

------------------------------------------------------------------------

# 11. Cluster Visualisation {#11-cluster-visualisation}

The project uses:

``` r
fviz_cluster(
  kmeans_model,
  data = rfm_scaled,
  geom = "point"
)
```

The supplied cluster plot shows:

-   Dim1: **55.5%**
-   Dim2: **30.2%**

Together, these displayed dimensions account for **85.7%** of the
variation represented in the two-dimensional projection.

Cluster 3 is visibly separated from the other clusters, while Clusters 1
and 2 are positioned more closely together.

------------------------------------------------------------------------

# 12. Cluster Evaluation {#12-cluster-evaluation}

The project creates a cluster profile using:

``` r
cluster_profile <- rfm %>%
  group_by(Cluster) %>%
  summarise(
    Customers = n(),
    AvgRecency = mean(Recency),
    AvgFrequency = mean(Frequency),
    AvgMonetary = mean(Monetary)
  )
```

Cluster numbers themselves have no business meaning. The interpretation
comes from the RFM characteristics of each group.

------------------------------------------------------------------------

# 13. Silhouette Analysis {#13-silhouette-analysis}

The silhouette score is calculated with:

``` r
sil <- silhouette(
  kmeans_model$cluster,
  dist(rfm_scaled)
)

mean(sil[, "sil_width"])
```

The supplied silhouette chart reports:

> **Average silhouette width: 0.59**

This indicates useful separation and cohesion among the three clusters.
Most observations have positive silhouette widths, although some
observations have weaker separation from neighbouring clusters.

The score should be interpreted together with the silhouette plot,
cluster plot, RFM profiles and business usefulness.

------------------------------------------------------------------------

# 14. Customer Segment Names {#14-customer-segment-names}

The project assigns:

``` r
rfm <- rfm %>%
  mutate(
    Segment = case_when(
      Cluster == 1 ~ "Dormant / Low-Value Customers",
      Cluster == 2 ~ "Core / Occasional Customers",
      Cluster == 3 ~ "VIP / High-Value Loyal Customers"
    )
  )
```

The three segments are:

1.  **Dormant / Low-Value Customers**
2.  **Core / Occasional Customers**
3.  **VIP / High-Value Loyal Customers**

These names are interpretations of the RFM profiles, not meanings
inherent in the cluster numbers.

------------------------------------------------------------------------

# 15. Final Segment Summary {#15-final-segment-summary}

The actual R output is:

  Segment                              Customers   Percentage
  ---------------------------------- ----------- ------------
  Core / Occasional Customers              3,225        74.3%
  Dormant / Low-Value Customers            1,087        25.1%
  VIP / High-Value Loyal Customers            26       0.599%

### Interpretation

The customer base is dominated by Core / Occasional Customers,
representing **74.3%** of customers.

Dormant / Low-Value Customers represent **25.1%**.

VIP / High-Value Loyal Customers make up only **0.599%**, or 26
customers.

The VIP segment is therefore extremely small, but its RFM profile shows
that it behaves very differently from the other groups.

------------------------------------------------------------------------

# 16. Final Segment Profile {#16-final-segment-profile}

The actual `final_profile` output is:

  Segment                              Customers   Avg Recency   Avg Frequency   Avg Monetary   Total Revenue
  ---------------------------------- ----------- ------------- --------------- -------------- ---------------
  Core / Occasional Customers              3,225          41.8             4.7          1,858       5,991,230
  Dormant / Low-Value Customers            1,087         247.0             1.6            632         686,665
  VIP / High-Value Loyal Customers            26           6.5            66.4         85,904       2,233,513

This table is the central business output of the project.

------------------------------------------------------------------------

# 17. Answers to the Final Project Questions {#17-answers-to-the-final-project-questions}

## How many customers are in each segment?

There are:

-   **3,225 Core / Occasional Customers**
-   **1,087 Dormant / Low-Value Customers**
-   **26 VIP / High-Value Loyal Customers**

The Core / Occasional segment is by far the largest group.

------------------------------------------------------------------------

## Which segment purchases most recently?

The **VIP / High-Value Loyal Customers** purchase most recently.

Their average Recency is **6.5 days**, compared with:

-   Core / Occasional: **41.8 days**
-   Dormant / Low-Value: **247.0 days**

Because lower Recency means a more recent purchase, the VIP segment is
the most recently active group.

------------------------------------------------------------------------

## Which segment purchases most frequently?

The **VIP / High-Value Loyal Customers** purchase most frequently.

Their average Frequency is **66.4 orders**, compared with:

-   Core / Occasional: **4.7**
-   Dormant / Low-Value: **1.6**

The difference shows that VIP customers are substantially more frequent
purchasers.

------------------------------------------------------------------------

## Which segment has the highest average spend?

The **VIP / High-Value Loyal Customers** have the highest average
Monetary value:

**85,904**

Compared with:

-   Core / Occasional: **1,858**
-   Dormant / Low-Value: **632**

This makes the VIP segment clearly distinct in terms of average customer
value.

------------------------------------------------------------------------

## Which segment generates the largest total revenue?

The **Core / Occasional Customers** generate the largest total revenue:

**5,991,230**

Compared with:

-   VIP / High-Value Loyal: **2,233,513**
-   Dormant / Low-Value: **686,665**

This is an important distinction. VIP customers have much higher average
value, but there are only 26 of them. Core / Occasional Customers
generate more total revenue because they represent 3,225 customers.

Therefore, **highest average customer value and highest total segment
revenue are not the same thing**.

------------------------------------------------------------------------

## Are any segments particularly small or large?

Yes.

The largest segment is **Core / Occasional Customers**, with:

**3,225 customers (74.3%)**

The smallest is **VIP / High-Value Loyal Customers**, with:

**26 customers (0.599%)**

Dormant / Low-Value Customers account for:

**1,087 customers (25.1%)**

The VIP segment is especially interesting because less than 1% of
customers account for **2,233,513** in total revenue.

------------------------------------------------------------------------

## How well separated are the clusters according to the silhouette score?

The average silhouette width is:

**0.59**

This provides useful evidence that the three clusters have meaningful
cohesion and separation.

The silhouette chart also shows mostly positive silhouette widths, while
a smaller number of observations have weaker separation.

The cluster plot supports the result visually, especially because
Cluster 3 is clearly separated from the other two groups in the
displayed projection.

------------------------------------------------------------------------

## What marketing action should the company take for each segment?

### Dormant / Low-Value Customers {#dormant--low-value-customers}

**Profile:** 1,087 customers, 247.0 average Recency, 1.6 average
Frequency, 632 average Monetary.

These customers are the least recently active and have the lowest
average purchasing activity and value.

**Recommended actions:**

-   Re-engagement emails
-   Purchase reminders
-   Targeted incentives
-   Limited promotional offers
-   Lower-cost marketing approaches
-   Campaigns designed to encourage a new purchase

The objective is to reactivate customers and move suitable customers
toward the Core segment.

### Core / Occasional Customers {#core--occasional-customers}

**Profile:** 3,225 customers, 41.8 average Recency, 4.7 average
Frequency, 1,858 average Monetary.

This is the largest segment and the largest contributor to total
revenue.

**Recommended actions:**

-   Encourage repeat purchases
-   Cross-selling
-   Upselling
-   Product recommendations
-   Loyalty incentives
-   Personalised promotions

Because this group represents 74.3% of customers, increasing engagement
even modestly could affect a large part of the customer base.

### VIP / High-Value Loyal Customers {#vip--high-value-loyal-customers}

**Profile:** 26 customers, 6.5 average Recency, 66.4 average Frequency,
85,904 average Monetary.

This is a very small but exceptionally high-value group.

**Recommended actions:**

-   VIP loyalty benefits
-   Personalised customer service
-   Exclusive offers
-   Early access to products or promotions
-   Personalised recommendations
-   Retention-focused communication

The objective should be to protect the relationship with these
high-value customers.

------------------------------------------------------------------------

# 18. Key Business Insights {#18-key-business-insights}

### Most customers are Core / Occasional {#most-customers-are-core--occasional}

At **74.3%**, this group represents the majority of customers.

### A quarter of customers are Dormant / Low-Value {#a-quarter-of-customers-are-dormant--low-value}

The Dormant group contains **1,087 customers (25.1%)** and has an
average Recency of **247 days**.

### VIP customers are rare but highly valuable

Only **26 customers** are in the VIP segment, yet they have:

-   **6.5 days average Recency**
-   **66.4 average Frequency**
-   **85,904 average Monetary**
-   **2,233,513 total revenue**

### The largest segment generates the most total revenue

Core / Occasional Customers generate **5,991,230** in total revenue.

This illustrates why a business should consider both **average customer
value** and **total segment revenue** when planning customer strategies.

------------------------------------------------------------------------

# 19. Important Concepts Demonstrated {#19-important-concepts-demonstrated}

### Unsupervised learning

There is no target variable containing the correct segment. K-means
discovers groups from the data.

### K-means

K-means groups customers according to their distance from cluster
centres.

### Centroids

A centroid represents the mean position of customers belonging to a
cluster.

### Scaling

Scaling prevents variables with larger numerical ranges from dominating
the distance calculation.

### `nstart`

`nstart` controls the number of different starting configurations. It is
not the number of clusters.

### `iter.max` {#itermax}

`iter.max` is the maximum number of adjustment iterations for each
starting configuration.

### Silhouette score

The silhouette score evaluates how well observations fit their own
cluster compared with neighbouring clusters.

### Cluster labels

Cluster numbers are arbitrary. Cluster 1 is not inherently better or
worse than Cluster 2 or Cluster 3.

------------------------------------------------------------------------

# 20. Project Workflow {#20-project-workflow}

``` text
Business Problem
       ↓
Load Data
       ↓
Inspect Data
       ↓
Clean Data
       ↓
Calculate Revenue
       ↓
Explore Data
       ↓
Create RFM Features
       ↓
Select RFM Variables
       ↓
Scale Variables
       ↓
Investigate Number of Clusters
       ↓
Run K-means
       ↓
Evaluate Clusters
       ↓
Profile Clusters
       ↓
Name Customer Segments
       ↓
Develop Business Recommendations
```

------------------------------------------------------------------------

# 21. Portfolio Checklist {#21-portfolio-checklist}

-   [x] Business problem defined
-   [x] Dataset described
-   [x] Data inspected
-   [x] Missing/problematic data cleaned
-   [x] Revenue calculated
-   [x] Customer behaviour explored
-   [x] RFM features created
-   [x] RFM distributions visualised
-   [x] RFM variables selected
-   [x] Variables scaled
-   [x] Elbow analysis performed
-   [x] Silhouette analysis performed
-   [x] K-means model created
-   [x] Reproducible seed used
-   [x] Clusters visualised
-   [x] Cluster sizes calculated
-   [x] Cluster profile created
-   [x] Silhouette score calculated
-   [x] Business-friendly segment names created
-   [x] Segment summary created
-   [x] Final segment profile created
-   [x] Business recommendations developed

------------------------------------------------------------------------

# 22. Limitations {#22-limitations}

### RFM captures only three dimensions

The model uses Recency, Frequency and Monetary value. It does not
include product preferences, demographics, marketing interactions or
other customer attributes.

### K-means is distance-based

Different clustering methods could potentially produce different
customer groups.

### RFM variables are skewed

The Frequency and Monetary distributions are strongly right-skewed. A
future version could investigate transformations such as logarithmic
transformation before clustering.

### VIP segment is very small

Only 26 customers belong to the VIP group. Its small size should be
considered when interpreting the overall customer base.

### Cluster names are interpretations

The names describe the observed RFM profiles. They are not labels
supplied by the original dataset.

------------------------------------------------------------------------

# 23. Possible Future Improvements {#23-possible-future-improvements}

Future versions could:

1.  Compare several K values more explicitly.
2.  Test alternative clustering algorithms.
3.  Investigate transformations for skewed variables.
4.  Add more customer-level features.
5.  Analyse the products purchased by each segment.
6.  Examine country-level differences between segments.
7.  Track how customers move between segments over time.
8.  Build an interactive dashboard.
9.  Connect segments to actual marketing campaign results.

------------------------------------------------------------------------

# 24. Suggested GitHub Repository Structure {#24-suggested-github-repository-structure}

``` text
customer-segmentation/
│
├── README.md
├── Customer_segmentation.R
├── Online Retail.xlsx
│
└── images/
    ├── Recency.png
    ├── Frequency.png
    ├── Monetary.png
    ├── elbow_plot.png
    ├── silhouette_score.png
    └── cluster_plot.png
```

------------------------------------------------------------------------

# 25. Conclusion {#25-conclusion}

This project demonstrates an end-to-end customer segmentation workflow
using **RFM analysis and K-means clustering**.

The analysis transforms raw e-commerce transactions into customer-level
behavioural information:

**Raw transactions → Data cleaning → Revenue → RFM → Scaling → K-means →
Evaluation → Customer segments → Business recommendations**

The final segmentation contains:

-   **3,225 Core / Occasional Customers (74.3%)**
-   **1,087 Dormant / Low-Value Customers (25.1%)**
-   **26 VIP / High-Value Loyal Customers (0.599%)**

The VIP segment is particularly notable. Although it contains less than
1% of customers, its members have an average Recency of **6.5 days**,
average Frequency of **66.4 orders**, and average Monetary value of
**85,904**.

The Core / Occasional segment generates the largest total revenue at
**5,991,230**, largely because it contains the majority of customers.

The model\'s **average silhouette width of 0.59** provides useful
evidence that the three clusters have meaningful separation and
cohesion.

The most important lesson is that clustering is not finished when an
algorithm produces cluster numbers. The clusters must be profiled and
translated into meaningful customer behaviour before they can support
business decisions.

This project therefore combines **data preparation, exploratory
analysis, RFM feature engineering, unsupervised machine learning, model
evaluation and business interpretation** into a complete data science
workflow.

------------------------------------------------------------------------

## Author

**Jaiyeoba Temitayo Felix**

**E-commerce Customer Segmentation Project**

Built with **R \| RFM Analysis \| K-means Clustering**
