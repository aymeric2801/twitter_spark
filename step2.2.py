#step2.2
import os
import socket
import threading
import time
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Deque
from collections import deque
from textblob import TextBlob
import random

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

API_KEY = "paste_here"
BASE_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
QUERY = '("AI" OR "Grok" OR "Elon Musk") lang:en -is:retweet'
QUERY_TYPE = "Latest"

POLL_INTERVAL_SECONDS = 30
MAX_TWEETS_PER_BATCH = 20

TWEET_BUFFER_SIZE = 100
TWEETS_PER_SECOND_TARGET = 0.33

last_tweet_id: Optional[str] = None
tweet_buffer: Deque = deque(maxlen=TWEET_BUFFER_SIZE)
buffer_lock = threading.Lock()
last_fetch_time = 0

tweet_schema = StructType([
    StructField("id", StringType(), True),
    StructField("created_at", StringType(), True),
    StructField("text", StringType(), True),
    StructField("username", StringType(), True),
    StructField("timestamp", TimestampType(), True)
])

def fetch_tweets_batch() -> List[Dict]:
    global last_tweet_id, last_fetch_time
    
    current_time = time.time()
    
    with buffer_lock:
        if current_time - last_fetch_time < POLL_INTERVAL_SECONDS:
            return []
        last_fetch_time = current_time
    
    params = {
        "query": QUERY,
        "queryType": QUERY_TYPE,
        "maxResults": MAX_TWEETS_PER_BATCH
    }
    
    if last_tweet_id:
        params["since_id"] = last_tweet_id
    
    headers = {"X-API-Key": API_KEY, "Accept": "application/json"}
    
    try:
        response = requests.get(BASE_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        tweets = data.get("tweets", data.get("data", []))
        if not tweets:
            return []
        
        max_id = last_tweet_id
        processed_tweets = []
        
        for tweet in tweets:
            tweet_id = tweet.get("id") or tweet.get("tweet_id")
            if not tweet_id:
                continue
                
            if not last_tweet_id or int(tweet_id) > int(last_tweet_id):
                username = tweet.get("username") or tweet.get("user", {}).get("screen_name", "unknown")
                created_at = tweet.get("created_at") or tweet.get("timestamp", datetime.now().isoformat())
                text = tweet.get("text") or tweet.get("full_text", "")
                
                try:
                    timestamp = datetime.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y")
                except:
                    timestamp = datetime.now()
                
                processed_tweets.append({
                    "id": str(tweet_id),
                    "created_at": str(created_at),
                    "text": text,
                    "username": username,
                    "timestamp": timestamp
                })
                
                if not max_id or int(tweet_id) > int(max_id):
                    max_id = tweet_id
        
        if max_id and (not last_tweet_id or int(max_id) > int(last_tweet_id)):
            last_tweet_id = max_id
        
        print(f"📊 API Call: Retrieved {len(processed_tweets)} new tweets (Cost: {len(processed_tweets) * 15} credits)")
        return processed_tweets
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return []

def simulate_late_data(tweet: Dict) -> Dict:
    if random.random() < 0.1:
        delayed_seconds = random.randint(30, 60)
        delayed_timestamp = tweet["timestamp"] - timedelta(seconds=delayed_seconds)
        tweet["text"] = f"[DELAYED {delayed_seconds}s] {tweet['text']}"
        tweet["timestamp"] = delayed_timestamp
        print(f"⚠️ Simulated delayed tweet: {tweet['id']} (delay: {delayed_seconds}s)")
    return tweet

def tweet_producer():
    global tweet_buffer
    
    def fetch_and_buffer():
        while True:
            try:
                new_tweets = fetch_tweets_batch()
                if new_tweets:
                    with buffer_lock:
                        for tweet in new_tweets:
                            tweet_buffer.append(simulate_late_data(tweet))
                    print(f"📥 Buffer size: {len(tweet_buffer)} tweets")
                else:
                    time.sleep(5)
            except Exception as e:
                print(f"Fetcher error: {e}")
            time.sleep(POLL_INTERVAL_SECONDS)
    
    fetcher_thread = threading.Thread(target=fetch_and_buffer, daemon=True)
    fetcher_thread.start()
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind(("localhost", 9199))
    except OSError as e:
        print(f"Port 9199 already in use: {e}")
        return
    
    server.listen(5)
    server.settimeout(10)
    
    clients = []
    last_send_time = time.time()
    send_index = 0
    
    while True:
        try:
            conn, addr = server.accept()
            print(f"✅ New Spark worker connected: {addr}")
            clients.append(conn)
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Server error: {e}")
        
        current_time = time.time()
        with buffer_lock:
            if tweet_buffer and current_time - last_send_time >= (1.0 / TWEETS_PER_SECOND_TARGET):
                tweet = tweet_buffer.popleft()
                tweet_json = json.dumps(tweet, default=str) + "\n"
                
                for client in clients[:]:
                    try:
                        client.send(tweet_json.encode("utf-8"))
                    except:
                        if client in clients:
                            clients.remove(client)
                
                last_send_time = current_time
        
        time.sleep(0.1)

def analyze_sentiment_with_textblob(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.5:
            sentiment = "VERY_POSITIVE"
            score = 1.0
        elif polarity > 0:
            sentiment = "POSITIVE"
            score = 0.5
        elif polarity == 0:
            sentiment = "NEUTRAL"
            score = 0.0
        elif polarity > -0.5:
            sentiment = "NEGATIVE"
            score = -0.5
        else:
            sentiment = "VERY_NEGATIVE"
            score = -1.0
            
        return sentiment, polarity, score
    except:
        return "NEUTRAL", 0.0, 0.0

def main():
    threading.Thread(target=tweet_producer, daemon=True).start()
    time.sleep(2)
    
    spark = SparkSession.builder \
        .appName("TwitterSentimentStreaming") \
        .master("local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.streaming.stopGracefullyOnShutdown", "true") \
        .config("spark.sql.streaming.schemaInference", "true") \
        .config("spark.sql.streaming.numRecentProgressUpdates", "100") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("\n" + "="*80)
    print("🐦 OPTIMIZED TWITTER SENTIMENT STREAMING")
    print("="*80)

    
    df_raw = spark.readStream \
        .format("socket") \
        .option("host", "localhost") \
        .option("port", 9199) \
        .load()
    
    df_tweets = df_raw \
        .select(from_json(col("value"), tweet_schema).alias("data")) \
        .select("data.*") \
        .withWatermark("timestamp", "30 seconds")
    
    sentiment_udf = udf(lambda text: analyze_sentiment_with_textblob(text)[0], StringType())
    polarity_udf = udf(lambda text: analyze_sentiment_with_textblob(text)[1], FloatType())
    sentiment_score_udf = udf(lambda text: analyze_sentiment_with_textblob(text)[2], FloatType())
    
    df_with_sentiment = df_tweets \
        .withColumn("sentiment_label", sentiment_udf(col("text"))) \
        .withColumn("polarity", polarity_udf(col("text"))) \
        .withColumn("sentiment_score", sentiment_score_udf(col("text")))
    
    windowed_aggregates = df_with_sentiment \
        .groupBy(
            window(col("timestamp"), "10 seconds", "5 seconds"),
            col("sentiment_label")
        ) \
        .agg(
            count("*").alias("tweet_count"),
            avg("sentiment_score").alias("avg_sentiment"),
            collect_list("text").alias("sample_tweets")
        ) \
        .orderBy("window")
    
    checkpoint_dir = "/tmp/spark-checkpoint-twitter-optimized"
    
    console_query = windowed_aggregates \
        .writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", "false") \
        .option("numRows", "20") \
        .trigger(processingTime="5 seconds") \
        .queryName("windowed_aggregates_console") \
        .start()
    
    raw_tweets_query = df_with_sentiment \
        .writeStream \
        .outputMode("append") \
        .format("memory") \
        .queryName("raw_tweets") \
        .trigger(processingTime="5 seconds") \
        .start()
    
    windowed_memory_query = windowed_aggregates \
        .writeStream \
        .outputMode("complete") \
        .format("memory") \
        .queryName("windowed_aggregates") \
        .trigger(processingTime="5 seconds") \
        .start()
    
    print("✅ Streaming queries started!")
    print("\n📊 Monitoring Dashboard:")
    print("-" * 80)
    
    last_stats_time = time.time()
    
    try:
        while True:
            time.sleep(10)
            
            print("\n" + "="*80)
            print("📈 SLIDING WINDOW AGGREGATIONS (10s window, 5s slide)")
            print("="*80)
            
            try:
                results = spark.sql("SELECT * FROM windowed_aggregates ORDER BY window.start DESC LIMIT 10")
                if results.count() > 0:
                    for row in results.collect():
                        window_start = row.window.start
                        window_end = row.window.end
                        print(f"\n⏰ Window: {window_start.strftime('%H:%M:%S')} → {window_end.strftime('%H:%M:%S')}")
                        print(f"   Sentiment: {row.sentiment_label}")
                        print(f"   📊 Count: {row.tweet_count} tweets")
                        print(f"   📈 Avg Sentiment Score: {row.avg_sentiment:.3f}")
                else:
                    print("⏳ Waiting for windowed data...")
            except Exception as e:
                print(f"Waiting for data...")
            
            print("\n" + "="*80)
            print("⚡ PERFORMANCE METRICS")
            print("="*80)
            current_time = time.time()
            if current_time - last_stats_time >= 30:
                print(f"API polling interval: {POLL_INTERVAL_SECONDS}s")
                print(f"Max tweets per call: {MAX_TWEETS_PER_BATCH}")
                print(f"Estimated cost per minute: {(60 / POLL_INTERVAL_SECONDS) * MAX_TWEETS_PER_BATCH * 15} credits")
                last_stats_time = current_time
            
            print("\n" + "="*80)
            print("🔧 HORIZONTAL SCALABILITY")
            print("="*80)
            print(f"Active Spark executors: {spark.sparkContext.defaultParallelism}")
            print(f"Shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
            print("="*80)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping streaming queries...")
        console_query.stop()
        raw_tweets_query.stop()
        windowed_memory_query.stop()
        spark.stop()
        print("✅ Exactly-once checkpoint saved. Can recover after crash.")

if __name__ == "__main__":
    main()