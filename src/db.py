import sqlite3
import json
from datetime import datetime, timedelta

DB_PATH = 'data/churn_predictions.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            input_data TEXT,
            model_used TEXT,
            prediction INTEGER,
            probability REAL,
            risk_category TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction(input_data: dict, model_used: str, prediction: int, probability: float, risk_category: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (timestamp, input_data, model_used, prediction, probability, risk_category)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (datetime.now(), json.dumps(input_data), model_used, prediction, probability, risk_category))
    conn.commit()
    conn.close()

def get_analytics():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Risk Distribution
    cursor.execute('SELECT risk_category, COUNT(*) FROM predictions GROUP BY risk_category')
    risk_dist = [{"name": row[0], "value": row[1]} for row in cursor.fetchall()]
    
    # Model Usage
    cursor.execute('SELECT model_used, COUNT(*) FROM predictions GROUP BY model_used')
    model_usage = [{"name": row[0], "value": row[1]} for row in cursor.fetchall()]
    
    # Total Predictions
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total_preds = cursor.fetchone()[0]
    
    # Time Series (Last 7 Days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Group by Date
    cursor.execute('''
        SELECT date(timestamp), COUNT(*) 
        FROM predictions 
        WHERE timestamp >= ?
        GROUP BY date(timestamp)
        ORDER BY date(timestamp)
    ''', (start_date.strftime('%Y-%m-%d'),))
    
    trend_data = [{"date": row[0], "predictions": row[1]} for row in cursor.fetchall()]

    conn.close()
    
    return {
        "risk_distribution": risk_dist,
        "model_usage": model_usage,
        "total_predictions": total_preds,
        "trend_data": trend_data
    }

if __name__ == '__main__':
    init_db()
