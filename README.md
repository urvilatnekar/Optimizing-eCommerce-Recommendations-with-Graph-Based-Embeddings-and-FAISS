# Optimizing-eCommerce-Recommendations-with-Graph-Based-Embeddings-and-FAISS

# 1]Understanding the Business Problem

Aim : Improve user experience by providing personalized product recommendations and diversifying the platform’s offerings, which includes recommending long-tail products.

Why is this needed? Personalization increases customer engagement, which boosts sales. Recommending diverse products (especially long-tail products) helps users discover new or niche items that they might not find through regular searches.

## Why a Graph-Based Recommender?

Improved personalization: Graph embeddings capture complex relationships between products that traditional methods like collaborative filtering may miss.

Platform diversity: By modeling user journeys as a graph, we can recommend long-tail products based on subtle behavior patterns.

Scalability: Techniques like FAISS make this approach scalable even with millions of products and users.

# 2]Data Collection and Data Understanding

Data Source: Logs from Flipkart capturing user events such as Product Page (PDP) views, Add-to-Cart (ATC) actions, and Purchases.

Attributes: Each row in the dataset has 9 attributes representing a user action. Key attributes include:
User ID: Identifies the user who performed the action.
Product ID: Identifies the product associated with the event.
Event Type: Specifies the type of action (PDP View, Add-to-Cart, Purchase).
Timestamp: When the event occurred.

Why is this important?

Understanding the data structure is critical before building a graph. Each action by a user gives insights into their intent, and you can map this intent to sequences of product interactions. For instance, a user who views several phones before making a purchase provides a journey we can map as a sequence of nodes and edges in a graph.

# 3]Why Use Graphs for Recommendations?

The nature of user interactions: In an eCommerce setting, users often exhibit specific behaviors:

Sequential interactions: A user might view several products in a sequence, which forms a chain of interactions.

Related products: Products that are viewed together by many users are often related in terms of features, categories, or pricing.

### Why a graph?

Graphs excel at representing relationships. In this context, nodes (products) and edges (interactions between products) provide a natural way to model user journeys. A graph allows us to capture relationships between products based on user behavior in a way that traditional collaborative filtering methods (e.g., matrix factorization) cannot.

Benefit: With a graph, the recommendation system can discover not only popular products but also less popular items (long-tail) based on the users' historical journey and interactions with similar products.

# 4]Graph Construction

Node: Each product in the dataset becomes a node.

Edge: An edge is created between two products when they are interacted with sequentially by the same user. 

For example:
User views product A, then product B, and adds product B to the cart. We create an edge between A and B.
Edge Direction: Directed edges are used because the order in which the user interacted with the products matters. If a user first views product A, then product B, the edge points from A to B.

## Why directed edges?
The direction of interaction gives us more context about user behavior. The user may first explore cheaper products before moving to more expensive ones, or they might research several items and then return to a previously viewed product to make a purchase.

## Why constructing the graph this way?

This approach captures intent and context of the user's journey. A simple view of product co-occurrence in purchases misses out on how users arrived at their final product.

# 5] Step 5: Graph Embeddings (DeepWalk/Node2Vec)

Graph Embedding: The goal is to convert each node (product) into a vector representation (embedding) that captures its relationships with other products in the graph.

DeepWalk and Node2Vec: Both are algorithms that generate embeddings by simulating random walks across the graph:

DeepWalk: Takes random walks from each node and treats each walk as a sequence (like a sentence in natural language processing). These sequences are used to train a model like Word2Vec to create embeddings.

Node2Vec: Extends DeepWalk by allowing you to control whether the random walks favor breadth-first (BFS) or depth-first (DFS) search strategies. This control helps adjust how much the embedding captures local vs. global relationships.

## Why use graph embeddings?

Capturing similarity: Products that are often viewed or purchased together will have similar embeddings, making it easy to recommend products with a vector-based similarity search.

### DeepWalk vs. Node2Vec: 

Node2Vec gives more control over the type of relationships captured, making it more flexible for building embeddings tailored to the recommendation context.

### How it works:

Graph Walks: For each product (node), perform multiple random walks of fixed length. These walks simulate user journeys across the product space.
Word2Vec Training: Treat each walk as a sentence and each product as a word. Train a Word2Vec model to learn embeddings that place related products close together in vector space.

# 6]Embedding Storage

Save the embeddings in a parquet format for easy retrieval and integration into the recommendation pipeline.

## Why parquet?

Efficiency: Parquet is a columnar storage format optimized for fast retrieval and large datasets. It's suitable for handling millions of product embeddings, enabling efficient queries during recommendation generation.

# 7]Step 7: Dimensionality Reduction (UMAP) and Visualization

UMAP (Uniform Manifold Approximation and Projection): Reduces the high-dimensional embeddings to 2D or 3D for visualization purposes.

## Why UMAP?

Interpretability: High-dimensional embeddings are difficult to interpret directly. By reducing them to two or three dimensions, you can visualize clusters of similar products.

Cluster Validation: Visualizing these clusters helps verify that similar products are indeed grouped together based on user interactions.

# 8]Similarity Search with FAISS

FAISS (Facebook AI Similarity Search): Used to perform fast similarity searches on the product embeddings.

Once you have embeddings for each product, you need a way to efficiently find the most similar products based on a user's interaction history. FAISS is highly optimized for this task.

How it works:

Indexing: FAISS builds an index of the product embeddings, allowing for fast retrieval of nearest neighbors (similar products) based on cosine or Euclidean distance.
Querying: For each product a user interacts with, FAISS finds similar products by searching for the nearest neighbors in the embedding space.

## Why FAISS?

Scalability: With millions of products, searching for recommendations in real-time would be computationally expensive using brute-force methods. FAISS uses advanced indexing techniques to enable fast and scalable similarity searches, making it suitable for large-scale eCommerce platforms.

# 9]Building the Recommendation Pipeline

Input: A user’s event history (viewed or purchased products).

Process:

Convert the product interactions into graph-based embeddings using Node2Vec or DeepWalk.

Use FAISS to search for the nearest neighbors (i.e., most similar products based on embeddings).

Output: A list of recommended products that are similar to the products the user has interacted with.

## Why this approach?

Intent-based recommendations: Instead of relying solely on what’s popular, this method recommends products based on the user’s journey and intent. This helps uncover long-tail products that may be of interest but aren’t widely popular.
