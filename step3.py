import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
import socket
import json
import os
import requests
from textblob import TextBlob
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, FloatType, TimestampType

st.set_page_config(
    page_title="Twitter Sentiment Dashboard",
    page_icon="🐦",
    layout="wide"
)

st.markdown("""
<style>
    .sentiment-positive { color: #28a745; font-weight: bold; }
    .sentiment-negative { color: #dc3545; font-weight: bold; }
    .sentiment-neutral { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 4g --executor-memory 4g pyspark-shell"

API_KEY = "paste_here"
BASE_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
QUERY = '("AI" OR "Grok" OR "Elon Musk") lang:en -is:retweet'
QUERY_TYPE = "Latest"
POLL_INTERVAL_SECONDS = 5
MAX_TWEETS_PER_BATCH = 20

tweet_cache = []
last_fetch_time = 0
fetch_lock = threading.Lock()

tweet_schema = StructType([
    StructField("id", StringType(), True),
    StructField("created_at", StringType(), True),
    StructField("text", StringType(), True),
    StructField("username", StringType(), True),
    StructField("timestamp", TimestampType(), True)
])

def fetch_tweets():
    global last_fetch_time, tweet_cache
    
    current_time = time.time()
    
    with fetch_lock:
        if current_time - last_fetch_time < POLL_INTERVAL_SECONDS:
            return list(tweet_cache)[-50:]
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
            return list(tweet_cache)[-50:]
        
        new_tweets = []
        for tweet in tweets[:MAX_TWEETS_PER_BATCH]:
            tweet_id = tweet.get("id") or tweet.get("tweet_id")
            if tweet_id:
                username = tweet.get("username") or tweet.get("user", {}).get("screen_name", "unknown")
                created_at = tweet.get("created_at") or tweet.get("timestamp", datetime.now().isoformat())
                text = tweet.get("text") or tweet.get("full_text", "")
                
                try:
                    timestamp = datetime.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y")
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
        
        if len(tweet_cache) > 1000:
            tweet_cache = tweet_cache[-1000:]
        
        return new_tweets
        
    except Exception as e:
        print(f"Error fetching tweets: {e}")
        return list(tweet_cache)[-50:]

def start_socket_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind(("localhost", 9199))
    except OSError as e:
        print(f"Port 9199 already in use: {e}")
        return

    server.listen(5)
    server.settimeout(2)
    print("Twitter producer listening on localhost:9199")
    
    clients = []
    tweet_buffer = []
    last_send_time = time.time()
    
    def fetch_and_buffer():
        nonlocal tweet_buffer, last_send_time
        while True:
            try:
                new_tweets = fetch_tweets()
                
                if new_tweets:
                    tweet_buffer.extend(new_tweets)
                    if len(tweet_buffer) > 200:
                        tweet_buffer = tweet_buffer[-200:]
                
                current_time = time.time()
                if tweet_buffer and (current_time - last_send_time > 0.5 or len(tweet_buffer) > 50):
                    for client in clients[:]:
                        try:
                            for tweet in tweet_buffer[-50:]:
                                tweet_json = json.dumps(tweet, default=str) + "\n"
                                client.send(tweet_json.encode("utf-8"))
                        except:
                            if client in clients:
                                clients.remove(client)
                    tweet_buffer = []
                    last_send_time = current_time
                
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error in fetch_and_buffer: {e}")
                time.sleep(1)
    
    fetch_thread = threading.Thread(target=fetch_and_buffer, daemon=True)
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

def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.5:
            sentiment = "VERY_POSITIVE"
            score = 1.0
        elif polarity > 0.2:
            sentiment = "POSITIVE"
            score = 0.6
        elif polarity > -0.2:
            sentiment = "NEUTRAL"
            score = 0.0
        elif polarity > -0.5:
            sentiment = "NEGATIVE"
            score = -0.6
        else:
            sentiment = "VERY_NEGATIVE"
            score = -1.0
            
        return sentiment, polarity, score
    except:
        return "NEUTRAL", 0.0, 0.0

class TwitterStreamingDashboard:
    def __init__(self):
        self.spark = None
        self.raw_tweets_query = None
        self.windowed_memory_query = None
        self.is_running = False
        self.start_time = None
        
    def initialize_spark(self):
        if self.spark is None:
            self.spark = SparkSession.builder \
                .appName("TwitterSentimentStreaming") \
                .master("local[2]") \
                .config("spark.driver.host", "127.0.0.1") \
                .config("spark.driver.bindAddress", "127.0.0.1") \
                .config("spark.sql.shuffle.partitions", "4") \
                .getOrCreate()
            
            self.spark.sparkContext.setLogLevel("ERROR")
        return self.spark
    
    def start_streaming(self):
        if self.is_running:
            return
        
        self.initialize_spark()
        self.start_time = time.time()
        
        threading.Thread(target=start_socket_server, daemon=True).start()
        time.sleep(1)
        
        df_raw = self.spark.readStream \
            .format("socket") \
            .option("host", "localhost") \
            .option("port", 9199) \
            .load()
        
        df_tweets = df_raw \
            .select(from_json(col("value"), tweet_schema).alias("data")) \
            .select("data.*") \
            .withWatermark("timestamp", "15 seconds")
        
        sentiment_udf = udf(lambda text: analyze_sentiment(text)[0], StringType())
        polarity_udf = udf(lambda text: analyze_sentiment(text)[1], FloatType())
        sentiment_score_udf = udf(lambda text: analyze_sentiment(text)[2], FloatType())
        
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
            .orderBy("window", ascending=False)
        
        self.raw_tweets_query = df_with_sentiment \
            .writeStream \
            .outputMode("append") \
            .format("memory") \
            .queryName("raw_tweets") \
            .trigger(processingTime="2 seconds") \
            .start()
        
        self.windowed_memory_query = windowed_aggregates \
            .writeStream \
            .outputMode("complete") \
            .format("memory") \
            .queryName("windowed_aggregates") \
            .trigger(processingTime="2 seconds") \
            .start()
        
        self.is_running = True
    
    def get_recent_tweets(self, limit=50):
        if not self.is_running:
            return pd.DataFrame()
        
        try:
            five_minutes_ago = datetime.now() - timedelta(minutes=5)
            df = self.spark.sql(f"""
                SELECT username, text, sentiment_label, polarity, 
                       sentiment_score, timestamp
                FROM raw_tweets 
                WHERE timestamp >= '{five_minutes_ago}'
                ORDER BY timestamp DESC 
                LIMIT {limit}
            """)
            return df.toPandas()
        except Exception as e:
            print(f"Error getting recent tweets: {e}")
            return pd.DataFrame()
    
    def get_tweets_last_5_minutes(self):
        if not self.is_running:
            return pd.DataFrame()
        
        try:
            five_minutes_ago = datetime.now() - timedelta(minutes=5)
            df = self.spark.sql(f"""
                SELECT username, text, sentiment_label, polarity, 
                       sentiment_score, timestamp
                FROM raw_tweets 
                WHERE timestamp >= '{five_minutes_ago}'
                ORDER BY timestamp DESC
            """)
            return df.toPandas()
        except Exception as e:
            print(f"Error getting tweets last 5 minutes: {e}")
            return pd.DataFrame()
    
    def get_windowed_stats(self):
        if not self.is_running:
            return pd.DataFrame()
        
        try:
            five_minutes_ago = datetime.now() - timedelta(minutes=5)
            df = self.spark.sql(f"""
                SELECT window.start as window_start, 
                       window.end as window_end,
                       sentiment_label, 
                       tweet_count, 
                       avg_sentiment
                FROM windowed_aggregates 
                WHERE window.start >= '{five_minutes_ago}'
                ORDER BY window.start DESC
                LIMIT 30
            """)
            return df.toPandas()
        except Exception as e:
            print(f"Error getting windowed stats: {e}")
            return pd.DataFrame()
    
    def get_overall_stats(self):
        if not self.is_running:
            return {}
        
        try:
            five_minutes_ago = datetime.now() - timedelta(minutes=5)
            df = self.spark.sql(f"""
                SELECT 
                    COUNT(*) as total_tweets,
                    AVG(sentiment_score) as avg_sentiment,
                    sentiment_label,
                    COUNT(*) as count
                FROM raw_tweets 
                WHERE timestamp >= '{five_minutes_ago}'
                GROUP BY sentiment_label
            """)
            stats_df = df.toPandas()
            
            if not stats_df.empty:
                total = stats_df['total_tweets'].iloc[0] if len(stats_df) > 0 else 0
                avg = stats_df['avg_sentiment'].iloc[0] if len(stats_df) > 0 else 0
                sentiment_counts = dict(zip(stats_df['sentiment_label'], stats_df['count']))
                
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    tweets_per_second = total / elapsed if elapsed > 0 else 0
                else:
                    tweets_per_second = 0
                
                return {
                    'total_tweets': total,
                    'avg_sentiment': avg,
                    'sentiment_counts': sentiment_counts,
                    'tweets_per_second': tweets_per_second
                }
        except Exception as e:
            print(f"Error getting overall stats: {e}")
        
        return {}
    
    def stop_streaming(self):
        if self.is_running:
            if self.raw_tweets_query:
                self.raw_tweets_query.stop()
            if self.windowed_memory_query:
                self.windowed_memory_query.stop()
            if self.spark:
                self.spark.stop()
            self.is_running = False

def main():
    st.title("🐦 Twitter Sentiment Analysis")
    
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = TwitterStreamingDashboard()
    
    with st.sidebar:
        if not st.session_state.dashboard.is_running:
            if st.button("Start Streaming", type="primary"):
                with st.spinner("Starting streaming engine..."):
                    st.session_state.dashboard.start_streaming()
                st.success("Streaming started!")
                st.rerun()
        else:
            if st.button("Stop Streaming", type="secondary"):
                st.session_state.dashboard.stop_streaming()
                st.success("Streaming stopped!")
                st.rerun()
        
        st.divider()
        
        st.markdown("### Sentiment Legend")
        st.markdown("""
        - 🟢 **VERY_POSITIVE** (> 0.5)
        - 🟢 **POSITIVE** (0.2 to 0.5)
        - 🟡 **NEUTRAL** (-0.2 to 0.2)
        - 🔴 **NEGATIVE** (-0.5 to -0.2)
        - 🔴 **VERY_NEGATIVE** (< -0.5)
        """)
    
    if st.session_state.dashboard.is_running:
        placeholder = st.empty()
        
        with placeholder.container():
            tweets_last_5min_df = st.session_state.dashboard.get_tweets_last_5_minutes()
            windowed_stats_df = st.session_state.dashboard.get_windowed_stats()
            overall_stats = st.session_state.dashboard.get_overall_stats()
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Tweets (5min)", overall_stats.get('total_tweets', 0))
            
            with col2:
                avg_sentiment = overall_stats.get('avg_sentiment', 0)
                st.metric("Avg Sentiment (5min)", f"{avg_sentiment:.3f}")
            
            with col3:
                tps = overall_stats.get('tweets_per_second', 0)
                st.metric("Tweets/sec", f"{tps:.1f}")
            
            with col4:
                if 'sentiment_counts' in overall_stats:
                    pos_count = overall_stats['sentiment_counts'].get('POSITIVE', 0) + overall_stats['sentiment_counts'].get('VERY_POSITIVE', 0)
                    st.metric("Positive (5min)", pos_count)
                else:
                    st.metric("Positive (5min)", 0)
            
            with col5:
                if 'sentiment_counts' in overall_stats:
                    neg_count = overall_stats['sentiment_counts'].get('NEGATIVE', 0) + overall_stats['sentiment_counts'].get('VERY_NEGATIVE', 0)
                    st.metric("Negative (5min)", neg_count)
                else:
                    st.metric("Negative (5min)", 0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Distribution (Last 5 Minutes)")
                if not tweets_last_5min_df.empty:
                    sentiment_counts = tweets_last_5min_df['sentiment_label'].value_counts()
                    fig = px.pie(
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Last 5 Minutes",
                        color_discrete_map={
                            'VERY_POSITIVE': '#28a745',
                            'POSITIVE': '#20c997',
                            'NEUTRAL': '#ffc107',
                            'NEGATIVE': '#fd7e14',
                            'VERY_NEGATIVE': '#dc3545'
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No tweets in last 5 minutes...")
            
            with col2:
                st.subheader("Sentiment Over Time (Last 5 Minutes)")
                if not tweets_last_5min_df.empty:
                    tweets_last_5min_df['timestamp'] = pd.to_datetime(tweets_last_5min_df['timestamp'])
                    time_series = tweets_last_5min_df.groupby(
                        tweets_last_5min_df['timestamp'].dt.floor('10S')
                    )['sentiment_score'].mean().reset_index()
                    
                    fig = px.line(
                        time_series,
                        x='timestamp',
                        y='sentiment_score',
                        title="Last 5 Minutes Evolution",
                        labels={'sentiment_score': 'Sentiment Score', 'timestamp': 'Time'}
                    )
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.update_layout(xaxis_title="Time", yaxis_title="Average Sentiment Score")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No sentiment data in last 5 minutes...")
            
            st.subheader("Windowed Aggregations (10s windows)")
            if not windowed_stats_df.empty:
                windowed_stats_df['window_start'] = pd.to_datetime(windowed_stats_df['window_start'])
                
                fig = px.bar(
                    windowed_stats_df,
                    x='window_start',
                    y='tweet_count',
                    color='sentiment_label',
                    title="Tweet Count by Sentiment (Last 5 Minutes)",
                    labels={'tweet_count': 'Tweet Count', 'window_start': 'Window Start'},
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Waiting for windowed data...")
            
            st.subheader(f"Recent Tweets (Last 5 Minutes - {len(tweets_last_5min_df)} tweets)")
            if not tweets_last_5min_df.empty:
                sentiment_emoji = {
                    'VERY_POSITIVE': '🟢',
                    'POSITIVE': '🟢',
                    'NEUTRAL': '🟡',
                    'NEGATIVE': '🔴',
                    'VERY_NEGATIVE': '🔴'
                }
                
                for idx, row in tweets_last_5min_df.head(30).iterrows():
                    time_diff = datetime.now() - pd.to_datetime(row['timestamp'])
                    minutes_ago = int(time_diff.total_seconds() / 60)
                    seconds_ago = int(time_diff.total_seconds() % 60)
                    
                    with st.expander(f"{sentiment_emoji.get(row['sentiment_label'], '⚪')} @{row['username']} - {row['sentiment_label']} (Score: {row['sentiment_score']:.2f})"):
                        st.write(row['text'])
                        if minutes_ago > 0:
                            st.caption(f"{minutes_ago}m {seconds_ago}s ago")
                        else:
                            st.caption(f"{seconds_ago}s ago")
            else:
                st.info("No tweets in last 5 minutes...")
        
        time.sleep(2)
        st.rerun()
    
    else:
        st.info("Click 'Start Streaming' to begin real-time sentiment analysis")

if __name__ == "__main__":
    main()