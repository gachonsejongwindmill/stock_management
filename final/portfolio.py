import os
import logging
import pandas as pd
from glob import glob
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import time
import re
import sys
import subprocess
import json
import builtins

class PrintLogger:
    def __init__(self, logger):
        self.logger = logger
        self.original_print = builtins.print
        
    def __call__(self, *args, **kwargs):
        self.logger.info(' '.join(map(str, args)))
        self.original_print(*args, **kwargs)

def setup_logging():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'portfolio_maker_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    print_logger = PrintLogger(logger)
    builtins.print = print_logger
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

logger = setup_logging()
def setup_data_loading() -> Tuple[str, List[str]]:
    """Setup and verify data loading configuration."""
    logger.info("Starting data loading setup")
    
    folder_path = "./plots_sp500"
    logger.info(f"Looking for forecast files in: {folder_path}")
    
    csv_files = glob(os.path.join(folder_path, "*_forecast.csv"))
    
    if not csv_files:
        error_msg = f"No forecast files found in {folder_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    logger.info(f"Found {len(csv_files)} forecast files")
    return folder_path, csv_files

def load_forecast_data(csv_files: List[str]) -> List[pd.DataFrame]:
    """Load forecast data from CSV files."""
    logger.info("Starting to load forecast data")
    dfs = []
    
    for idx, file_path in enumerate(csv_files, 1):
        try:
            filename = os.path.basename(file_path)
            symbol = filename.replace("_forecast.csv", "")
            logger.debug(f"[{idx}/{len(csv_files)}] Processing {filename}, symbol: {symbol}")
            
            logger.debug(f"Reading CSV: {file_path}")
            df = pd.read_csv(file_path, usecols=["Date", "Predicted"])
            
            if df.empty:
                logger.warning(f"Empty DataFrame for {symbol}, skipping...")
                continue
                
            if 'Date' not in df.columns or 'Predicted' not in df.columns:
                logger.error(f"Required columns not found in {filename}")
                continue
            
            df.rename(columns={"Predicted": symbol}, inplace=True)
            
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            df = df.drop_duplicates(subset=['Date'], keep='last')
            
            dfs.append(df)
            logger.debug(f"Successfully loaded {len(df)} rows for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            continue
    
    if not dfs:
        error_msg = "No valid forecast data was loaded"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    logger.info(f"Successfully loaded {len(dfs)} out of {len(csv_files)} forecast files")
    return dfs

def merge_forecast_data(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Merge multiple forecast DataFrames on the 'Date' column."""
    logger.info("Starting to merge forecast data")
    
    if not dfs:
        error_msg = "No DataFrames to merge"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        logger.debug(f"Merging {len(dfs)} DataFrames")
        from functools import reduce
        merged_df = reduce(lambda left, right: pd.merge(left, right, on='Date', how='outer'), dfs)
        
        logger.debug("Sorting by date")
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        merged_df = merged_df.sort_values('Date')
        
        merged_df = merged_df.drop_duplicates(subset=['Date'], keep='last')
        
        logger.info(f"Merged DataFrame shape: {merged_df.shape}")
        logger.debug(f"Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
        logger.debug(f"Columns: {', '.join([col for col in merged_df.columns if col != 'Date'])}")
        
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging DataFrames: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        logger.info("=== Starting Portfolio Maker ===")
        
        folder_path, csv_files = setup_data_loading()
        dfs = load_forecast_data(csv_files)
        merged_df = merge_forecast_data(dfs)
        
        logger.info("Data loading and merging completed successfully")
        
    except Exception as e:
        logger.critical(f"Fatal error in Portfolio Maker: {str(e)}", exc_info=True)
        raise
    merged_df_2 = merged_df.copy()
    merged_df_2.iloc[:, 1:] = merged_df_2.iloc[:, 1:].pct_change()
    merged_df_2.dropna(inplace=True)
    std_series = merged_df_2.iloc[:, 1:].std()

    def classify_risk(std):
        if std >= 0.2:
            return "Very High Risk"
        elif 0.1 <= std < 0.2:
            return "High Risk"
        elif 0.06 <= std < 0.1:
            return "Medium Risk"
        elif 0.01 <= std < 0.06:
            return "Low Risk"
        else:
            return "Very Low Risk"

    risk_levels = std_series.apply(classify_risk)


    price_df = merged_df.iloc[:, 1:]

    first_prices = price_df.iloc[0]

    max_prices = price_df.max()
    min_prices = price_df.min()
    change_rates = ((max_prices - first_prices) / first_prices).round(4)
    change_rates2 = ((min_prices - first_prices) / first_prices).round(4)

    risk_df = risk_levels.reset_index()
    risk_df.columns = ['Stock', 'RiskLevel']

    change_df = pd.DataFrame({
        'Stock': change_rates.index,
        'Risk_Levels' : risk_df['RiskLevel'].values,
        'Max_ChangeRate': change_rates.values,
        'Min_ChangeRate': change_rates2.values
    })


    import subprocess
    import json

    def chat_with_gemma(message: str,temperature: float = 0.0) -> str:
        import requests
        
        url = "http://localhost:11434/api/generate"  
        payload = {
            "model": "gemma3:12b",
            "prompt": message,
            "stream": False,
            "temperature": temperature
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()  
            data = response.json()
            return data.get("response", "").strip()
            
        except requests.exceptions.RequestException as e:
            print(f"Error in chat_with_gemma (HTTP request failed): {str(e)}")
            return "" 

    def get_news_cache_file() -> str:
        cache_dir = os.path.join(os.path.dirname(__file__), 'news_cache')
        os.makedirs(cache_dir, exist_ok=True)
        today = datetime.now().strftime('%Y-%m-%d')
        return os.path.join(cache_dir, f'news_cache_{today}.csv')

    def load_cached_news() -> Dict[str, List[Dict]]:
        cache_file = get_news_cache_file()
        if not os.path.exists(cache_file):
            return {}
        
        try:
            df = pd.read_csv(cache_file)
            df['articles'] = df['articles'].apply(eval)
            return df.set_index('symbol')['articles'].to_dict()
        except Exception as e:
            print(f"Could not load news cache: {str(e)}")
            return {}

    def save_news_to_cache(symbol: str, articles: List[Dict]):
        try:
            cache_file = get_news_cache_file()
            
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file)
                mask = df['symbol'] == symbol
                if mask.any():
                    df.loc[mask, 'articles'] = str(articles)
                else:
                    df = pd.concat([df, pd.DataFrame([{'symbol': symbol, 'articles': str(articles)}])], 
                                ignore_index=True)
            else:
                df = pd.DataFrame([{'symbol': symbol, 'articles': str(articles)}])
            
            df.to_csv(cache_file, index=False)
        except Exception as e:
            print(f"Could not save news cache: {str(e)}")

    def get_stock_news(symbol: str, use_cache: bool = True, api_key: str = "afd7dddd5a0e4c13b8f504529f664e48") -> List[Dict]:
        if use_cache:
            cached_news = load_cached_news()
            if symbol in cached_news:
                print(f"   - {symbol} (from cache)", end='', flush=True)
                return cached_news[symbol]
        
        try:
            print(f"\n   - {symbol} (fetching news...)", end='', flush=True)
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)
            
            to_date_str = to_date.strftime('%Y-%m-%d')
            from_date_str = from_date.strftime('%Y-%m-%d')
            
            url = f"https://newsapi.org/v2/everything?q={symbol} stock&from={from_date_str}&to={to_date_str}&language=en&sortBy=publishedAt&apiKey={api_key}"
            print(f"\n   - API URL: {url.split('&apiKey=')[0]}...") 
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            print(f"   - API Response Status: {response.status_code}")
            print(f"   - Total Results: {data.get('totalResults', 0)}")
            
            if 'status' in data and data['status'] == 'error':
                print(f"   - API Error: {data.get('message', 'Unknown error')}")
                return []
                
            articles = data.get('articles', [])[:5]
            
            if not articles:
                print(f"   - No articles found for {symbol}")
                return []
                
            print(f"   - Found {len(articles)} articles for {symbol}")
            
            save_news_to_cache(symbol, articles)
            
            return articles
            
        except requests.exceptions.RequestException as e:
            print(f"\nRequest failed for {symbol}:")
            print(f"   - Error Type: {type(e).__name__}")
            print(f"   - Error Details: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   - Status Code: {e.response.status_code}")
                try:
                    error_data = e.response.json()
                    print(f"   - Error Message: {error_data.get('message', 'No error message')}")
                    print(f"   - Error Code: {error_data.get('code', 'N/A')}")
                except:
                    print(f"   - Response Text: {e.response.text[:200]}...")
        except Exception as e:
            print(f"\nUnexpected error for {symbol}:")
            print(f"   - Error Type: {type(e).__name__}")
            print(f"   - Error Details: {str(e)}")
        
        return []

    def analyze_news_sentiment(articles: List[Dict]) -> Dict:
        if not articles:
            return {"sentiment": "neutral", "summary": "No recent news articles found."}
        
        texts = [f"{article.get('title', '')}. {article.get('description', '')}" 
                for article in articles]
        
        combined_text = "\n".join(texts)
        
        prompt = f"""
        Analyze the sentiment of the following news articles about a company.
        Consider the overall tone (positive/negative/neutral) and provide a brief summary.
        
        News Articles:
        {combined_text}
        Return a JSON object in the following format:
        answer: [
        {{
            "sentiment": "positive",
            "summary": "The company reported strong quarterly earnings and announced new product launches.",
            "impact": "positive"
        }}
        ]

        Make sure your answer follows the same JSON structure exactly, replacing the example values with the analysis based on the news articles.

        """
        
        try:
            response = chat_with_gemma(prompt)
            match = re.search(r'answer:\s*(\[\s*\{.*?\}\s*\])', response, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                return data
        except Exception as e:
            print(f"Error analyzing news sentiment: {str(e)}")
        
        return {"sentiment": "neutral", "summary": "Could not analyze sentiment.", "impact": "neutral"}

    def split_stock_table(stock_table, max_tokens=4000):
        chunk_size = max_tokens * 2
        chunks = []
        current_chunk = []
        current_length = 0
        
        lines = stock_table.strip().split('\n')
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            if current_length + line_length > chunk_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(line)
            current_length += line_length
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    def analyze_stock_chunk(chunk, user_age, loss_tolerance_percent, chunk_num, total_chunks):
        analysis_prompt = f"""
        You are a financial analyst. Analyze the following stock data and identify the most promising candidates for a diversified portfolio.
        
        User profile:
        - Age: {user_age}
        - Loss tolerance: {loss_tolerance_percent}% of total assets
        
        This is chunk {chunk_num} of {total_chunks}.
        
        Stock data (S&P 500 stocks):
        {chunk}
        
        For each stock, consider:
        1. Risk level (Very High to Very Low)
        2. Predicted 21-day return
        3. Diversification across sectors
        
        Return a list of 3-5 stock symbols with brief reasoning for each selection.
        Focus on stocks with the best risk-return profile for the user's risk tolerance.
        
        Format your response with one stock symbol per line, optionally followed by a colon and reasoning.
        Example:
        AAPL: Strong fundamentals and growth potential
        GOOGL: Dominant in cloud computing
        
        Important: Only include the stock symbols and optional reasoning, nothing else.
        """
        
        print(f"Analyzing chunk {chunk_num} of {total_chunks}...")
        return chat_with_gemma(analysis_prompt)
    def extract_final_symbols(response_text):

        match = re.search(r'answer:\s*(\[\s*\{.*?\}\s*\])', response_text, re.DOTALL)
        if match:
            try:
                final_stocks = json.loads(match.group(1))
                symbols = [stock.get("symbol") for stock in final_stocks if "symbol" in stock]
                if symbols:
                    return symbols
            except json.JSONDecodeError:
                print("JSON 파싱 실패, fallback으로 텍스트 분석 시도")
        
        symbols = []
        for line in response_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if ':' in line:
                symbol = line.split(':')[0].strip()
            elif len(line) <= 5:  
                symbol = line
            else:
                continue
            if symbol and symbol not in symbols:
                symbols.append(symbol)
        
        return symbols
    def analyze_stocks(stock_table, user_age, loss_tolerance_percent):
       
        chunks = split_stock_table(stock_table)
        
        if len(chunks) == 1:
           
            print("Analyzing stocks to select top candidates...")
            initial_analysis = analyze_stock_chunk(chunks[0], user_age, loss_tolerance_percent, 1, 1)
        else:
            
            print(f"Processing {len(chunks)} chunks of stock data...")
            chunk_results = []
            for i, chunk in enumerate(chunks, 1):
                result = analyze_stock_chunk(chunk, user_age, loss_tolerance_percent, i, len(chunks))
                chunk_results.append(result)
                time.sleep(1)  
            
    
            initial_analysis = '\n'.join(chunk_results)
        
        print("Initial stock analysis complete. Selected symbols:")
        
        
      
        lines = [line.strip() for line in initial_analysis.split('\n') if line.strip()]
        top_symbols = []
        
        for line in lines:
     
            if ':' in line:
                symbol = line.split(':')[0].strip().replace('-', '').strip()
                if symbol and len(symbol) <= 5:  
                    top_symbols.append(symbol)
                    print(f"   - {symbol}: {line.split(':', 1)[1].strip() if ':' in line else ''}")
            elif len(line.split()) == 1 and 1 <= len(line) <= 5: 
                top_symbols.append(line.strip())
                print(f"   - {line.strip()}")
        
       
        top_symbols = list(dict.fromkeys(top_symbols))[:10]
        print(f"Fetching news for {len(top_symbols)} top stocks...")
        
        stock_news = {}
        

        cached_news = load_cached_news()
        
        for symbol in top_symbols:
            try:
                
                if symbol in cached_news and cached_news[symbol]:
                    articles = cached_news[symbol]
                    print(f"   - {symbol} (from cache)", end='')
                    from_cache = True
                else:
                   
                    print(f"   - {symbol} (fetching...)", end='')
                    articles = get_stock_news(symbol, use_cache=False)
                    from_cache = False
                
                if articles:
                    if not from_cache:
                        print(f" ✓ Found {len(articles)} articles")
                    
                    
                    print(f"   - Analyzing sentiment for {symbol}...", end='')
                    sentiment = analyze_news_sentiment(articles)
                    print(f" {sentiment['sentiment'].upper()}")
                    
                    stock_news[symbol] = {
                        'articles': [{'title': a.get('title', ''), 'url': a.get('url', '')} for a in articles],
                        'sentiment': sentiment
                    }
                    
                  
                    if articles and 'title' in articles[0]:
                        print(f"     Sample: {articles[0].get('title', 'No title')}")
                else:
                    print(f"   - {symbol}: No articles found")
                    stock_news[symbol] = {
                        'articles': [],
                        'sentiment': {'sentiment': 'neutral', 'summary': 'No recent news found.'}
                    }
                
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                stock_news[symbol] = {
                    'articles': [],
                    'sentiment': {'sentiment': 'error', 'summary': f'Error: {str(e)}'}
                }
            
            print() 
            time.sleep(1)  
        
        print("News Collection Summary:")
        for symbol, data in stock_news.items():
            article_count = len(data.get('articles', []))
            sentiment = data.get('sentiment', {}).get('sentiment', 'unknown').upper()
            print(f"   - {symbol}: {article_count} articles, {sentiment} sentiment")
        
        final_analysis_prompt = f"""
        Based on the initial stock analysis and recent news sentiment, refine the stock selection.

        Initial Analysis:
        {initial_analysis}

        Recent News Analysis:
        {json.dumps(stock_news, indent=2)}

        Update your stock recommendations considering both the financial metrics and news sentiment.
        Return a JSON object in the following format:

        answer: [
        {{
            "symbol": "AAPL",
            "reason": "Strong fundamentals and positive news sentiment"
        }},
        {{
            "symbol": "MSFT",
            "reason": "Strong fundamentals and positive news sentiment"
        }}
        ]

        Make sure to return 5-10 stock objects, each with "symbol" and "reason" keys.
        """

        print("Analyzing news sentiment and refining stock selection...")
        final_analysis = chat_with_gemma(final_analysis_prompt)

        final_symbols = extract_final_symbols(final_analysis)
        
        final_stock_news = {k: v for k, v in stock_news.items() if k in final_symbols}
        
        for symbol in final_symbols:
            if symbol not in final_stock_news:
                final_stock_news[symbol] = {
                    'articles': [],
                    'sentiment': {'sentiment': 'neutral', 'summary': 'No recent news found.'}
                }
        
        return final_stock_news

    def create_portfolio(analysis_results, user_context, attempt):
        allocation_prompt = f"""
        Based on the following analysis and user context, create an optimal portfolio allocation.
        
        User context:
        - Age: {user_context['age']}
        - Risk tolerance: {user_context['risk_tolerance']}%
        - Investment horizon: {user_context.get('horizon', 'medium-term')}
        
        Analysis results (including news sentiment):
        {analysis_results}
        
        Please provide a portfolio with 5 stocks that balances risk and return, considering both financial metrics and recent news sentiment.
        
        Guidelines:
        0. Only use stocks that are in the S&P500.
        1. Favor stocks with positive news sentiment and strong fundamentals.
        2. Be cautious with stocks that have negative news, even if their financials look good.
        3. Ensure proper diversification across sectors.
        4. Adjust sector and stock risk levels according to the user's age and risk tolerance:
        - Younger investors with higher risk tolerance may have more growth-oriented allocations.
        - Older investors or those with lower risk tolerance should prioritize stability and income.
        5. Ensure the overall portfolio risk profile aligns with the user's risk tolerance percentage.
        6. Include exactly 5 stocks in the portfolio.
        7. Return the portfolio in this exact JSON format:
        {{
            "stocks": ["SYM1", "SYM2", "SYM3", "SYM4", "SYM5"],
            "allocation": [0.3, 0.25, 0.2, 0.15, 0.1],
            "reasoning": "Brief explanation including how news sentiment influenced the allocation",
            "risk_analysis": "Brief assessment of the overall portfolio risk profile"
        }}
        """
        return chat_with_gemma(allocation_prompt, attempt*0.1)

    def validate_stocks_exist(recommended_stocks, available_stocks):
        missing = [s for s in recommended_stocks if s not in available_stocks]
        if missing:
            print(f"Warning: The following stocks are not in our data: {', '.join(missing)}")
        return not missing

    import re
    import json

    def parse_portfolio_response(response_str, max_retries=1):
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                match = re.search(r'\{[^{}]*\}', response_str, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    portfolio = json.loads(json_str)

                    if "stocks" in portfolio and isinstance(portfolio["stocks"], list):
                        portfolio["stocks"] = [
                            s.strip().upper() for s in portfolio["stocks"] if isinstance(s, str)
                        ]

                    if not all(key in portfolio for key in ['stocks', 'allocation', 'reasoning']):
                        raise ValueError("Missing required fields in portfolio response")
                        
                    if len(portfolio['stocks']) != 5 or len(portfolio['allocation']) != 5:
                        raise ValueError("Portfolio must contain exactly 5 stocks")
                        
                    if not all(isinstance(x, (int, float)) for x in portfolio['allocation']):
                        raise ValueError("Allocation values must be numbers")
                        
                    if abs(sum(portfolio['allocation']) - 1.0) > 0.01:
                        total = sum(portfolio['allocation'])
                        portfolio['allocation'] = [round(x/total, 4) for x in portfolio['allocation']]
                        
                    return portfolio
                    
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < max_retries:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    continue
                else:
                    print(f"Warning: Could not parse JSON response: {e}")
                    return 1004
            
            try:
                match = re.search(r'\{.*\}', response_str, re.DOTALL)
                if match:
                    dict_str = match.group(0)
                    parsed = eval(dict_str, {"__builtins__": None}, {})
                    if isinstance(parsed, dict):
                        stocks = [s.strip().upper() for s in parsed.keys()]
                        allocations = list(parsed.values())
                        return {
                            'stocks': stocks,
                            'allocation': allocations,
                            'reasoning': 'Generated from legacy format'
                        }
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"Attempt {attempt + 1} failed in legacy parsing: {e}. Retrying...")
                    continue
                else:
                    print(f"Warning: Could not parse legacy format: {e}")
                    return 1004
        

    def get_portfolio_with_retry(analysis_results, user_context, max_attempts=5):
        """Get portfolio with retry logic if recommended stocks are not found."""
        available_stocks = set(change_df['Stock'].tolist())
        for attempt in range(1, max_attempts + 1):
            print(f"Portfolio Generation Attempt {attempt}/{max_attempts}")
            
            portfolio_response = create_portfolio(analysis_results, user_context,attempt)
            print("Portfolio recommendation received.")
            
            try:
                portfolio = parse_portfolio_response(portfolio_response)
                if portfolio == 1004:
                    raise Exception("양식 불합격")
                if validate_stocks_exist(portfolio['stocks'], available_stocks):
                    return portfolio_response
                    
                print(f"Some recommended stocks are not available. "
                    f"Retrying with different stocks... (Attempt {attempt}/{max_attempts})")
                
            except Exception as e:
                print(f"Error parsing portfolio: {str(e)}")
                if attempt == max_attempts:
                    print("Max retry attempts reached. Using best available portfolio.")
                    return portfolio_response
        
        print("Failed to generate a valid portfolio after multiple attempts.")
        return None

    user_context = {
        'age': sys.argv[1],
        'risk_tolerance': sys.argv[2],
        'horizon': 'long-term',
        'news_api_key': 'afd7dddd5a0e4c13b8f504529f664e48' 
    }

    stock_table = change_df.to_string(index=False)

    print("Analyzing stock data...")
    analysis_results = analyze_stocks(stock_table, user_context['age'], user_context['risk_tolerance'])
    print("Analysis complete. Creating portfolio...")

    portfolio_response = get_portfolio_with_retry(analysis_results, user_context)

    if portfolio_response is None:
        print("Could not generate a valid portfolio. Please try again or check your data.")
        exit(1)


    portfolio = parse_portfolio_response(portfolio_response)
    print("Recommended Portfolio:")
    for stock, weight in zip(portfolio['stocks'], portfolio['allocation']):
        print(f"{stock}: {weight*100:.1f}%")

    print(f"Strategy: {portfolio['reasoning']}")

    result = dict(zip(portfolio['stocks'], portfolio['allocation']))

    stock_store = []
    valid_stocks = []

    for stock, weight in result.items():
        try:
            if stock not in merged_df.columns:
                print(f"Warning: {stock} not found in data. This should not happen with the retry mechanism!")
                continue
                
            max_price = merged_df[stock].max()
            std_return = std_series[stock]

            result[stock] = [round(weight, 4), round(max_price, 2), round(std_return, 4)]
            stock_store.append(stock)
            valid_stocks.append(stock)
            
        except KeyError as e:
            print(f"Error processing {stock}: {str(e)}")
            continue

    if not valid_stocks:
        print("Error: No valid stocks were found in the data.")
        exit(1)


    result = {k: v for k, v in result.items() if k in valid_stocks}
    result2= {}
    for stock in merged_df.columns:
        if stock == 'Date':
            continue

        stock_data = [
            {"y": round(value, 2), "ds": str(date)}
            for date, value in zip(merged_df["Date"], merged_df[stock])
            if pd.notna(value)
        ]

        result2[stock] = stock_data
    result2 = {k: v for k, v in result2.items() if k in stock_store}
    
    def make_result3_invest(result, result2, selected_stocks=None, initial=100):
        if selected_stocks is None:
            selected_stocks = list(result.keys())

        weights = {s: float(result[s][0]) for s in selected_stocks if s in result}
        total_w = sum(weights.values())
        print(total_w)
        if total_w == 0:
            n = len(weights)
            weights = {s: 1.0/n for s in weights}
        else:
            weights = {s: w/total_w for s,w in weights.items()}

        price_map = {}
        first_date_of_stock = {}
        for s in selected_stocks:
            entries = result2.get(s, [])
            date_price = {}
            for e in entries:
                ds = e["ds"]
                price = float(e["y"])
                date_price[ds] = price
            if date_price:
                sorted_dates = sorted(date_price.keys())
                first_date_of_stock[s] = sorted_dates[0]
                price_map[s] = {d: date_price[d] for d in sorted_dates}

        shares = {} 
        for s, ratio in weights.items():
            if s not in price_map:
                shares[s] = 0.0
                continue
            buy_date = first_date_of_stock[s]
            buy_price = price_map[s].get(buy_date)
            if not buy_price or buy_price == 0:
                shares[s] = 0.0
            else:
                allocation = initial * ratio
                shares[s] = allocation / buy_price

        all_dates = set()
        for s, pm in price_map.items():
            all_dates.update(pm.keys())
        sorted_all_dates = sorted(all_dates)

        last_known_price = {s: None for s in price_map}
        result3 = []
        for ds in sorted_all_dates:
            total_value = 0.0
            for s in price_map:
                if ds in price_map[s]:
                    last_known_price[s] = price_map[s][ds]
                price_for_day = last_known_price[s]
                if price_for_day is None:
                    continue
                total_value += shares.get(s, 0.0) * price_for_day

            result3.append({"y": round(total_value, 5), "ds": ds})  
        return result3

    result3 = make_result3_invest(result,result2)
    portfolio_reasoning = portfolio['reasoning']
    translation_prompt = f"""
    Translate the following investment analysis to Korean.

    **Instructions:**
    1. Maintain a professional and clear tone.
    2. Keep the financial terms and stock symbols in English.
    3. **Crucially: Do not write any introductory text, explanations, or summaries. The output must begin directly with the Korean translation and contain nothing else.**

    **Text to translate:**
    {portfolio_reasoning}
    """
    
    ollama_api_url = "http://localhost:11434/api/generate"

    ollama_payload = {
        "model": "gemma3:latest", 
        "prompt": translation_prompt.strip(),
        "stream": False 
    }

    try:
        response = requests.post(
            ollama_api_url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(ollama_payload)
        )
        
        response.raise_for_status()  
        response.encoding = 'utf-8'
        response_data = response.json()
        
        result4 = response_data.get('response', '번역 결과를 가져오지 못했습니다.').strip()
        


    except requests.exceptions.RequestException as e:
        error_message = f"Ollama API 연결에 실패했습니다: {str(e)}"
        print(error_message)
        result4 = portfolio_reasoning 
        
    except Exception as e:
        error_message = f"번역 중 알 수 없는 에러가 발생했습니다: {str(e)}"
        print(error_message)
        result4 = portfolio_reasoning 
    print(json.dumps({"result": result, "result2": result2,"result3":result3,"result4":result4}, ensure_ascii=False))
