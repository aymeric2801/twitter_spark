#step2
import os
import socket
import threading
import time
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from textblob import TextBlob
import hashlib
from collections import deque

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 2g --executor-memory 2g pyspark-shell"

API_KEY = "paste_here"
BASE_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
QUERY = '("AI" OR "Grok" OR "Elon Musk") lang:en -is:retweet'
QUERY_TYPE = "Latest"

POLL_INTERVAL_SECONDS = 15
MAX_TWEETS_PER_BATCH = 5

tweet_cache = deque(maxlen=1000)
last_fetch_time = 0
fetch_lock = threading.Lock()
last_tweet_ids = {}

tweet_schema = StructType([
    StructField("id", StringType(), True),
    StructField("created_at", StringType(), True),
    StructField("text", StringType(), True),
    StructField("username", StringType(), True),
    StructField("timestamp", TimestampType(), True)
])

def fetch_tweets_efficient():
    global last_fetch_time, tweet_cache, last_tweet_ids
    
    current_time = time.time()
    
    with fetch_lock:
        if current_time - last_fetch_time < POLL_INTERVAL_SECONDS:
            return []
        last_fetch_time = current_time
    
    params = {
        "query": QUERY, 
        "queryType": QUERY_TYPE,
        "maxResults": MAX_TWEETS_PER_BATCH
    }
    
    headers = {"X-API-Key": API_KEY, "Accept": "application/json"}
    
    try:
        response = requests.get(BASE_URL, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        tweets = data.get("tweets", data.get("data", []))
        if not tweets:
            return []
        
        new_tweets = []
        for tweet in tweets[:MAX_TWEETS_PER_BATCH]:
            tweet_id = tweet.get("id") or tweet.get("tweet_id")
            if not tweet_id:
                continue
                
            tweet_hash = hashlib.md5(f"{tweet_id}_{tweet.get('text', '')[:100]}".encode()).hexdigest()
            
            if tweet_hash in last_tweet_ids and time.time() - last_tweet_ids[tweet_hash] < 3600:
                continue
                
            username = tweet.get("username") or tweet.get("user", {}).get("screen_name", "unknown")
            created_at = tweet.get("created_at") or tweet.get("timestamp", datetime.now().isoformat())
            text = tweet.get("text") or tweet.get("full_text", "")
            
            try:
                if " +0000 " in created_at:
                    timestamp = datetime.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y")
                else:
                    timestamp = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except:
                timestamp = datetime.now()
            
            new_tweet = {
                "id": str(tweet_id),
                "created_at": str(created_at),
                "text": text,
                "username": username,
                "timestamp": timestamp
            }
            
            new_tweets.append(new_tweet)
            tweet_cache.append(new_tweet)
            last_tweet_ids[tweet_hash] = time.time()
            
            if len(last_tweet_ids) > 10000:
                current = time.time()
                last_tweet_ids = {k: v for k, v in last_tweet_ids.items() if current - v < 3600}
        
        return new_tweets
        
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return []

def optimized_socket_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind(("localhost", 9199))
    except OSError as e:
        print(f"Port 9199 already in use: {e}")
        return

    server.listen(5)
    server.settimeout(2)
    print(f"Twitter producer listening on localhost:9199 (polling every {POLL_INTERVAL_SECONDS}s)")
    
    clients = []
    tweet_buffer = []
    last_send_time = time.time()
    last_batch_time = time.time()
    
    def fetch_and_cache():
        nonlocal tweet_buffer, last_send_time, last_batch_time
        
        while True:
            try:
                new_tweets = fetch_tweets_efficient()
                
                if new_tweets:
                    tweet_buffer.extend(new_tweets)
                    
                    cost_per_request = MAX_TWEETS_PER_BATCH * 15
                
                current_time = time.time()
                if tweet_buffer and (current_time - last_send_time > 1 or len(tweet_buffer) > 20):
                    for client in clients[:]:
                        try:
                            batch = tweet_buffer[:50]
                            batch_json = "\n".join([json.dumps(t, default=str) for t in batch])
                            client.send((batch_json + "\n").encode("utf-8"))
                        except Exception as e:
                            print(f"Socket error: {e}")
                            if client in clients:
                                clients.remove(client)
                    
                    tweet_buffer = []
                    last_send_time = current_time
                
                if len(new_tweets) == 0:
                    time.sleep(POLL_INTERVAL_SECONDS)
                else:
                    time.sleep(POLL_INTERVAL_SECONDS)
                
            except Exception as e:
                print(f"Error in fetch_and_cache: {e}")
                time.sleep(5)
    
    fetch_thread = threading.Thread(target=fetch_and_cache, daemon=True)
    fetch_thread.start()
    
    while True:
        try:
            conn, addr = server.accept()
            print(f"New Spark worker connected: {addr}")
            clients.append(conn)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"Server error: {e}")
            time.sleep(1)

def analyze_sentiment_with_textblob(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.5:
            sentiment = "🔴 VERY POSITIVE"
            emoji = "😍"
        elif polarity > 0:
            sentiment = "🟢 POSITIVE"
            emoji = "🙂"
        elif polarity == 0:
            sentiment = "⚪ NEUTRAL"
            emoji = "😐"
        elif polarity > -0.5:
            sentiment = "🟡 NEGATIVE"
            emoji = "😕"
        else:
            sentiment = "🔴 VERY NEGATIVE"
            emoji = "😡"
            
        return sentiment, polarity, emoji
    except:
        return "⚪ NEUTRAL", 0.0, "😐"

def main():
    threading.Thread(target=optimized_socket_server, daemon=True).start()
    time.sleep(2)
    
    spark = SparkSession.builder \
        .appName("TwitterSentimentStreaming") \
        .master("local[1]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.sql.shuffle.partitions", "1") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.streaming.backpressure.enabled", "true") \
        .config("spark.streaming.kafka.maxRatePerPartition", "10") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    print("\n" + "="*60)
    print("🐦 TWITTER SENTIMENT STREAMING (COST OPTIMIZED)")
    print("="*60 + "\n")
    
    df_raw = spark.readStream \
        .format("socket") \
        .option("host", "localhost") \
        .option("port", 9199) \
        .load()
    
    df_tweets = df_raw \
        .select(from_json(col("value"), tweet_schema).alias("data")) \
        .select("data.*")
    
    sentiment_udf = udf(lambda text: analyze_sentiment_with_textblob(text)[0], StringType())
    polarity_udf = udf(lambda text: analyze_sentiment_with_textblob(text)[1], FloatType())
    emoji_udf = udf(lambda text: analyze_sentiment_with_textblob(text)[2], StringType())
    
    df_with_sentiment = df_tweets \
        .withColumn("sentiment", sentiment_udf(col("text"))) \
        .withColumn("polarity", polarity_udf(col("text"))) \
        .withColumn("emoji", emoji_udf(col("text")))
    
    def process_row(df, epoch_id):
        count = df.count()
        if count == 0:
            return
        
        rows = df.collect()
        for row in rows:
            print("\n" + "="*60)
            print(f"📝 NEW TWEET")
            print("="*60)
            print(f"👤 @{row.username}")
            print(f"📅 {row.created_at[:19] if row.created_at else 'N/A'}")
            print(f"💬 {row.text}")
            print("-"*60)
            print(f"{row.emoji} Sentiment: {row.sentiment}")
            print(f"📊 Polarity Score: {row.polarity:.4f} (-1=negative, +1=positive)")
            print("="*60)
    
    query = df_with_sentiment \
        .writeStream \
        .foreachBatch(process_row) \
        .trigger(processingTime="10 seconds") \
        .start()
    
    print("✅ Streaming started! Waiting for tweets...")
    print(f"💡 Cost optimization active: {MAX_TWEETS_PER_BATCH} tweets per request")
    print(f"💡 Polling every {POLL_INTERVAL_SECONDS} seconds")
    print("\n")
    
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping streaming...")
        query.stop()
        spark.stop()
        print("✅ Done")

if __name__ == "__main__":
    main()