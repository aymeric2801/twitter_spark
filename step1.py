#step1
import requests
import time
from datetime import datetime
import json
from typing import List, Dict, Optional
from collections import deque
import threading
from config import API_KEY, BASE_URL, QUERY, QUERY_TYPE

MAX_TWEETS_PER_REQUEST = 20  # Fixed by API
BATCH_SIZE = 5  # Process in smaller batches
COOLDOWN_THRESHOLD = 10  # If fewer than this many tweets, increase interval
MIN_POLL_INTERVAL = 5
MAX_POLL_INTERVAL = 60
ADAPTIVE_POLLING = True

class OptimizedTweetFetcher:
    def __init__(self):
        self.last_tweet_id: Optional[str] = None
        self.tweet_cache = deque(maxlen=1000)  # Store up to 1000 tweets
        self.consecutive_low_tweets = 0
        self.current_interval = MIN_POLL_INTERVAL
        self.lock = threading.Lock()
        self.session = requests.Session()  # Reuse connection
        self.session.headers.update({
            "X-API-Key": API_KEY,
            "Accept": "application/json"
        })
        
    def fetch_tweets(self) -> List[Dict]:
        """Fetch tweets with optimized credit usage"""
        with self.lock:
            params = {"query": QUERY, "queryType": QUERY_TYPE}
            if self.last_tweet_id:
                params["since_id"] = self.last_tweet_id
            
            try:
                # Use session for connection pooling
                response = self.session.get(BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                tweets = data.get("tweets", data.get("data", []))
                if not tweets:
                    return []
                
                # Process tweets efficiently
                new_tweets = []
                max_id = self.last_tweet_id
                
                for tweet in tweets:
                    tweet_id = tweet.get("id") or tweet.get("tweet_id")
                    if not tweet_id:
                        continue
                    
                    # Only process new tweets
                    if self.last_tweet_id and int(tweet_id) <= int(self.last_tweet_id):
                        continue
                    
                    processed = {
                        "id": tweet_id,
                        "created_at": tweet.get("created_at") or tweet.get("timestamp"),
                        "text": tweet.get("text") or tweet.get("full_text", ""),
                        "username": tweet.get("username") or tweet.get("user", {}).get("screen_name", "unknown"),
                        "fetched_at": datetime.now().isoformat()
                    }
                    new_tweets.append(processed)
                    self.tweet_cache.append(processed)
                    
                    if not max_id or int(tweet_id) > int(max_id):
                        max_id = tweet_id
                
                if max_id:
                    self.last_tweet_id = max_id
                
                # Update adaptive polling
                if ADAPTIVE_POLLING:
                    if len(new_tweets) < COOLDOWN_THRESHOLD:
                        self.consecutive_low_tweets += 1
                        # Increase interval exponentially with each consecutive low batch
                        self.current_interval = min(
                            MAX_POLL_INTERVAL,
                            MIN_POLL_INTERVAL * (2 ** (self.consecutive_low_tweets // 3))
                        )
                    else:
                        self.consecutive_low_tweets = 0
                        self.current_interval = MIN_POLL_INTERVAL
                
                return new_tweets
                
            except requests.exceptions.RequestException as e:
                print(f"API error: {e}")
                # On error, increase interval to avoid hammering API
                self.current_interval = min(self.current_interval * 2, MAX_POLL_INTERVAL)
                return []
            except Exception as e:
                print(f"Unexpected error: {e}")
                return []
    
    def get_tweets_since(self, since_id: Optional[str] = None) -> List[Dict]:
        """Fetch tweets since specific ID - useful for catching up"""
        old_last_id = self.last_tweet_id
        if since_id:
            self.last_tweet_id = since_id
        try:
            return self.fetch_tweets()
        finally:
            if since_id:
                self.last_tweet_id = old_last_id
    
    def get_recent_tweets(self, limit: int = 50) -> List[Dict]:
        """Get recent tweets from cache without API call"""
        return list(self.tweet_cache)[-limit:]

def batch_processor(tweets: List[Dict], batch_size: int = BATCH_SIZE):
    """Process tweets in batches to spread out work"""
    for i in range(0, len(tweets), batch_size):
        yield tweets[i:i + batch_size]

def main():
    fetcher = OptimizedTweetFetcher()
    print(f"Adaptive polling enabled - interval: {MIN_POLL_INTERVAL}-{MAX_POLL_INTERVAL}s")
    print(f"Initial interval: {MIN_POLL_INTERVAL}s")
    
    last_print_time = time.time()
    print_interval = 30  # Print summary every 30 seconds
    
    while True:
        try:
            start_time = time.time()
            tweets = fetcher.fetch_tweets()
            fetch_time = time.time() - start_time
            
            if tweets:
                # Process tweets in batches to distribute work
                for batch in batch_processor(tweets):
                    for t in batch:
                        print(f"[{t.get('created_at', '?')}] @{t.get('username', '?')}: {t.get('text', '')[:80]}...")
            
            # Periodic summary
            if time.time() - last_print_time >= print_interval:
                total_tweets = len(fetcher.tweet_cache)
                print(f"\n--- Summary ---")
                print(f"Total tweets in cache: {total_tweets}")
                print(f"Current poll interval: {fetcher.current_interval}s")
                print(f"Last fetch: {len(tweets)} tweets in {fetch_time:.2f}s")
                print(f"Credits used this fetch: {len(tweets) * 15} (since {len(tweets)} tweets)")
                print(f"Cache size: {len(fetcher.tweet_cache)}/1000")
                print("----------------\n")
                last_print_time = time.time()
            
            # Wait with adaptive interval
            sleep_time = max(0, fetcher.current_interval - fetch_time)
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()