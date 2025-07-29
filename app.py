import os
import base64
import uuid
import sqlite3
import json
import re
import logging
import random
import time
from datetime import datetime, timedelta
from flask import Flask, request, render_template, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import requests
from functools import wraps
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Android app
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'snapschart-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATABASE_PATH'] = 'trading_analyses.db'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Rate limiting dictionary
api_calls = {}
MAX_CALLS_PER_MINUTE = 10

class SimplifiedAIAnalyzer:
    def __init__(self):
        # Your API keys - use environment variables in production
        self.api_keys = {
            'together': os.getenv('TOGETHER_API_KEY'),
            'groq': os.getenv('GROQ_API_KEY')
        }
        
        # Professional trading analysis prompt
        self.analysis_prompt = """As a senior quantitative trading analyst with 15+ years of experience, provide comprehensive chart analysis in this EXACT format:

🎯 EXECUTIVE SUMMARY:
- Primary Signal: BUY/SELL/HOLD with specific bias
- Confidence Score: X% 
- Based on [key reasoning]
- Market Regime: TRENDING/RANGING/BREAKOUT with volatility assessment
- Risk Level: LOW/MEDIUM/HIGH

📊 TECHNICAL ANALYSIS:
- Chart Patterns: Identify specific patterns (channels, triangles, flags, etc.)
- Support Levels:
  * S1: [price] ([strength])
  * S2: [price] ([strength])
- Resistance Levels:
  * R1: [price] ([strength]) 
  * R2: [price] ([strength])
- Trend Analysis: Short-term and long-term trend direction and strength
- Volume Analysis: Volume patterns and conviction levels
- Technical Indicators: Moving averages, momentum, oscillators analysis

💰 TRADING SETUP:
- Entry Zone: [specific price range] for [long/short] positions
- Stop Loss: [price] ([risk percentage])
- Take Profit 1: [price] ([R:R ratio])
- Take Profit 2: [price] ([R:R ratio]) 
- Take Profit 3: [price] ([R:R ratio])
- Position Sizing: [percentage] of portfolio maximum due to [conditions]

⚡ EXECUTION STRATEGY:
- Entry Triggers: Wait for [specific trigger/confirmation]
- Time Frame: [execution timeframe] for entry, [trend timeframe] for direction
- Market Timing: Active during [specific session/hours]
- Risk Management: [specific risk management rules]

🧠 MARKET PSYCHOLOGY:
- Current Sentiment: [sentiment] with [bias]
- Key Psychological Levels: [round numbers and key levels]
- Institutional Activity: [possible institutional behavior]

⚠️ RISK ASSESSMENT:
- Trade Invalidation: [specific invalidation level/condition]
- Market Risks:
  * [risk factor 1]
  * [risk factor 2] 
  * [risk factor 3]
- Correlation Risks: [correlation analysis]

Additional Notes: [Any additional market context, distribution/accumulation signs, recommendations for different trader types]

IMPORTANT: Provide specific price levels, percentages, and actionable insights. Use concrete numbers and clear directional bias."""

    def encode_image(self, image_path):
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_with_groq(self, image_path):
        """Analyze chart using Groq API - 14,400 requests/day FREE"""
        try:
            base64_image = self.encode_image(image_path)
            
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_keys['groq']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",  # Working model
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.3
            }
            
            logger.info("🚀 Using Groq Llama 4 Scout (FREE - blazing fast!)")
            response = requests.post(url, headers=headers, json=payload, timeout=45)
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['choices'][0]['message']['content']
                return {
                    'success': True,
                    'api': 'groq',
                    'analysis': analysis,
                    'model': 'llama-4-scout-17b-16e-instruct',
                    'response_time': response.elapsed.total_seconds()
                }
            else:
                return {
                    'success': False,
                    'api': 'groq',
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            logger.error(f"Groq analysis error: {e}")
            return {
                'success': False,
                'api': 'groq',
                'error': str(e)
            }

    def analyze_with_together(self, image_path):
        """Analyze chart using Together AI - $25 FREE credits"""
        try:
            base64_image = self.encode_image(image_path)
            
            url = "https://api.together.xyz/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_keys['together']}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",  # Working model
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.analysis_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.3
            }
            
            logger.info("🚀 Using Together AI Llama Vision Turbo ($25 credits)")
            response = requests.post(url, headers=headers, json=payload, timeout=90)
            
            if response.status_code == 200:
                result = response.json()
                analysis = result['choices'][0]['message']['content']
                return {
                    'success': True,
                    'api': 'together',
                    'analysis': analysis,
                    'model': 'llama-3.2-11b-vision-turbo',
                    'response_time': response.elapsed.total_seconds()
                }
            else:
                return {
                    'success': False,
                    'api': 'together',
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            logger.error(f"Together AI analysis error: {e}")
            return {
                'success': False,
                'api': 'together',
                'error': str(e)
            }

    def get_random_analysis(self, image_path):
        """Randomly select between Groq and Together AI with fallback"""
        available_apis = [
            ('groq', self.analyze_with_groq),
            ('together', self.analyze_with_together)
        ]
        
        # Randomly shuffle the order
        random.shuffle(available_apis)
        
        for api_name, api_func in available_apis:
            if self.api_keys[api_name]:
                logger.info(f"🎲 Trying {api_name.upper()}")
                result = api_func(image_path)
                
                if result['success']:
                    logger.info(f"✅ {api_name.upper()} succeeded")
                    return result
                else:
                    logger.warning(f"❌ {api_name.upper()} failed: {result.get('error', 'Unknown error')}")
                    continue
        
        return {
            'success': False,
            'error': 'All AI services failed. Please try again later.'
        }
    
    def get_dual_analysis(self, image_path):
        """Get analysis from both Groq and Together AI for comparison"""
        groq_result = self.analyze_with_groq(image_path)
        together_result = self.analyze_with_together(image_path)
        
        return {
            'groq': groq_result,
            'together': together_result,
            'comparison_available': groq_result['success'] and together_result['success']
        }

# Initialize AI analyzer
multi_ai = SimplifiedAIAnalyzer()

def rate_limit(max_calls=MAX_CALLS_PER_MINUTE):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            now = time.time()
            client_ip = request.remote_addr
            
            if client_ip not in api_calls:
                api_calls[client_ip] = []
            
            # Remove calls older than 1 minute
            api_calls[client_ip] = [call_time for call_time in api_calls[client_ip] if now - call_time < 60]
            
            if len(api_calls[client_ip]) >= max_calls:
                return jsonify({'error': 'Rate limit exceeded. Please wait.'}), 429
            
            api_calls[client_ip].append(now)
            return f(*args, **kwargs)
        return wrapper
    return decorator

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    """Get database connection with proper error handling"""
    try:
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise

def init_database():
    """Initialize SQLite database for storing analyses"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Create analyses table
        c.execute('''CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            analysis_text TEXT NOT NULL,
            model_used TEXT DEFAULT 'groq',
            signal_type TEXT,
            confidence INTEGER,
            entry_price TEXT,
            stop_loss TEXT,
            take_profit TEXT,
            file_size INTEGER,
            processing_time REAL,
            ai_service TEXT,
            user_session TEXT
        )''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized")
        
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise

def cleanup_old_files():
    """Clean up uploaded files older than 7 days"""
    try:
        upload_folder = app.config['UPLOAD_FOLDER']
        cutoff_time = time.time() - (7 * 24 * 60 * 60)  # 7 days ago
        
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path) and os.path.getctime(file_path) < cutoff_time:
                os.remove(file_path)
                logger.info(f"Cleaned up old file: {filename}")
                
    except Exception as e:
        logger.error(f"File cleanup error: {e}")

def extract_trading_signals(analysis_text):
    """Extract structured trading signals from analysis"""
    signal_type = "NEUTRAL"
    confidence = 50
    entry_price = "Not specified"
    stop_loss = "Not specified"
    take_profit = "Not specified"
    
    if not analysis_text:
        return signal_type, confidence, entry_price, stop_loss, take_profit
    
    try:
        text_upper = analysis_text.upper()
        
        # Look for signals
        if any(phrase in text_upper for phrase in ["BUY", "BULLISH", "LONG"]):
            signal_type = "BUY"
        elif any(phrase in text_upper for phrase in ["SELL", "BEARISH", "SHORT"]):
            signal_type = "SELL"
        
        # Extract confidence
        confidence_match = re.search(r'CONFIDENCE[:\s]*(\d+)%', text_upper)
        if confidence_match:
            confidence = int(confidence_match.group(1))
        
        # Extract prices (simplified)
        entry_match = re.search(r'ENTRY[:\s]*[^0-9]*(\d+[,.-]*\d*)', analysis_text, re.IGNORECASE)
        if entry_match:
            entry_price = f"${entry_match.group(1)}"
        
        stop_match = re.search(r'STOP[:\s]*LOSS[:\s]*[^0-9]*(\d+[,.-]*\d*)', analysis_text, re.IGNORECASE)
        if stop_match:
            stop_loss = f"${stop_match.group(1)}"
        
        tp_match = re.search(r'TAKE[:\s]*PROFIT[:\s]*[^0-9]*(\d+[,.-]*\d*)', analysis_text, re.IGNORECASE)
        if tp_match:
            take_profit = f"${tp_match.group(1)}"
    
    except Exception as e:
        logger.error(f"Signal extraction error: {e}")
    
    return signal_type, confidence, entry_price, stop_loss, take_profit

def save_analysis_to_db(filename, analysis, signal_type, confidence, entry_price, 
                       stop_loss, take_profit, ai_service, model, file_size, 
                       response_time, user_session):
    """Save analysis to database"""
    analysis_id = str(uuid.uuid4())
    
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('''INSERT INTO analyses 
                    (id, timestamp, filename, analysis_text, signal_type, confidence, 
                     entry_price, stop_loss, take_profit, model_used, file_size, 
                     processing_time, ai_service, user_session)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (analysis_id, datetime.now().isoformat(), filename, analysis, 
                   signal_type, confidence, entry_price, stop_loss, take_profit, 
                   model, file_size, response_time, ai_service, user_session))
        conn.commit()
        conn.close()
        logger.info(f"Analysis saved to database with ID: {analysis_id}")
    except sqlite3.Error as e:
        logger.error(f"Database save error: {e}")
    
    return analysis_id

# ===== API ROUTES =====

@app.route('/')
def index():
    """Main page"""
    return jsonify({
        'message': 'SnapChart Trading Analysis API',
        'version': '2.0',
        'endpoints': {
            'test': '/api/test',
            'analyze': '/api/analyze-chart',
            'stats': '/api/stats'
        }
    })

@app.route('/api/test', methods=['GET'])
@rate_limit(max_calls=5)
def test_api():
    """Test AI APIs connection"""
    results = {}
    
    # Test Groq
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {multi_ai.api_keys['groq']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{"role": "user", "content": "Respond with: 'Groq API working'"}],
            "max_tokens": 50
        }
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            results['groq'] = {'status': 'SUCCESS', 'message': 'Groq API working'}
        else:
            results['groq'] = {'status': 'ERROR', 'message': f'HTTP {response.status_code}'}
    except Exception as e:
        results['groq'] = {'status': 'ERROR', 'message': str(e)}
    
    # Test Together AI
    try:
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {multi_ai.api_keys['together']}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            "messages": [{"role": "user", "content": "Respond with: 'Together AI working'"}],
            "max_tokens": 50
        }
        response = requests.post(url, headers=headers, json=data, timeout=10)
        
        if response.status_code == 200:
            results['together'] = {'status': 'SUCCESS', 'message': 'Together AI working'}
        else:
            results['together'] = {'status': 'ERROR', 'message': f'HTTP {response.status_code}'}
    except Exception as e:
        results['together'] = {'status': 'ERROR', 'message': str(e)}
    
    # Overall status
    working_apis = sum(1 for api in results.values() if api['status'] == 'SUCCESS')
    overall_status = 'SUCCESS' if working_apis > 0 else 'ERROR'
    
    return jsonify({
        'status': overall_status,
        'message': f'{working_apis}/2 AI APIs working',
        'apis': results,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze-chart', methods=['POST'])
@rate_limit(max_calls=10)
def api_analyze_chart():
    """API endpoint for chart analysis"""
    logger.info("=== API CHART ANALYSIS STARTED ===")
    start_time = time.time()
    
    try:
        # Check if file is present
        if 'chart_image' not in request.files and 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        # Get file
        file = request.files.get('chart_image') or request.files.get('file')
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Get parameters
        ai_service = request.form.get('ai_service', 'random')
        user_session = request.form.get('user_session', str(uuid.uuid4()))
        symbol = request.form.get('symbol', 'UNKNOWN')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        file_size = os.path.getsize(file_path)
        logger.info(f"File saved: {unique_filename}, Size: {file_size} bytes, AI: {ai_service}")
        
        # Get analysis based on requested AI service
        if ai_service == 'groq':
            result = multi_ai.analyze_with_groq(file_path)
        elif ai_service == 'together':
            result = multi_ai.analyze_with_together(file_path)
        elif ai_service == 'both':
            dual_result = multi_ai.get_dual_analysis(file_path)
            processing_time = time.time() - start_time
            
            response_data = {
                'success': True,
                'mode': 'dual',
                'groq_result': dual_result['groq'],
                'together_result': dual_result['together'],
                'comparison_available': dual_result['comparison_available'],
                'processing_time': round(processing_time, 2),
                'file_info': {
                    'filename': unique_filename,
                    'size': file_size,
                    'symbol': symbol
                },
                'timestamp': datetime.now().isoformat()
            }
            return jsonify(response_data)
        else:
            # Random selection (default)
            result = multi_ai.get_random_analysis(file_path)
        
        processing_time = time.time() - start_time
        
        if result['success']:
            # Extract trading signals
            signal_type, confidence, entry_price, stop_loss, take_profit = extract_trading_signals(result['analysis'])
            
            # Save to database
            analysis_id = save_analysis_to_db(
                unique_filename, result['analysis'], signal_type, confidence,
                entry_price, stop_loss, take_profit, result['api'], 
                result.get('model', result['api']), file_size, 
                result['response_time'], user_session
            )
            
            # Prepare response
            response_data = {
                'success': True,
                'analysis_id': analysis_id,
                'ai_service': result['api'],
                'model': result.get('model', result['api']),
                'analysis': result['analysis'],
                'signal_type': signal_type,
                'confidence': confidence,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'response_time': result['response_time'],
                'processing_time': round(processing_time, 2),
                'file_info': {
                    'filename': unique_filename,
                    'url': f'/uploads/{unique_filename}',
                    'size': file_size,
                    'symbol': symbol
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Analysis completed: {result['api'].upper()}, Signal: {signal_type}")
            return jsonify(response_data)
        else:
            # Clean up file on error
            try:
                os.remove(file_path)
            except:
                pass
            
            return jsonify({
                'success': False,
                'error': result['error'],
                'ai_service': result['api']
            }), 500
            
    except Exception as e:
        logger.error(f"❌ API Analysis error: {str(e)}")
        
        # Clean up file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint for getting trading statistics"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Get recent analyses (last 30 days)
        c.execute('''SELECT signal_type, confidence, processing_time, ai_service 
                     FROM analyses WHERE timestamp > ?''', 
                  [(datetime.now() - timedelta(days=30)).isoformat()])
        recent_analyses = c.fetchall()
        
        # Calculate statistics
        total = len(recent_analyses)
        buy_signals = len([a for a in recent_analyses if a['signal_type'] == 'BUY'])
        sell_signals = len([a for a in recent_analyses if a['signal_type'] == 'SELL'])
        neutral_signals = total - buy_signals - sell_signals
        
        # AI service usage
        groq_usage = len([a for a in recent_analyses if a['ai_service'] == 'groq'])
        together_usage = len([a for a in recent_analyses if a['ai_service'] == 'together'])
        
        # Average metrics
        avg_confidence = sum([a['confidence'] or 0 for a in recent_analyses]) / total if total > 0 else 0
        avg_processing_time = sum([a['processing_time'] or 0 for a in recent_analyses]) / total if total > 0 else 0
        
        conn.close()
        
        stats = {
            'total_analyses': total,
            'signals': {
                'buy': buy_signals,
                'sell': sell_signals,
                'neutral': neutral_signals
            },
            'ai_usage': {
                'groq': groq_usage,
                'together': together_usage
            },
            'averages': {
                'confidence': round(avg_confidence, 1),
                'processing_time': round(avg_processing_time, 2)
            },
            'period': '30 days',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({'success': True, 'stats': stats})
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== ERROR HANDLERS =====

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'success': False, 'error': 'Rate limit exceeded. Please wait before making another request.'}), 429

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'success': False, 'error': 'Internal server error. Please try again.'}), 500

if __name__ == '__main__':
    print("🚀 ChartAI - Multi-AI Trading Analysis Platform")
    print("🎨 Web Interface + Android API Backend")
    print("🤖 Powered by Groq + Together AI")
    print("💰 Advanced Trading Signal Extraction")
    print("🔒 Security & Rate Limiting Enabled")
    print("📱 Android App API Ready")
    print("🌐 Visit: http://localhost:8080")
    print("")
    
    # Initialize database
    try:
        init_database()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
    
    # Check API keys
    groq_key = os.getenv('GROQ_API_KEY')
    together_key = os.getenv('TOGETHER_API_KEY')
    
    print("📋 API Configuration:")
    if groq_key:
        print(f"✅ Groq API Key: {groq_key[:15]}...")
    else:
        print("⚠️  Groq API Key: Not configured")
        print("   Set with: export GROQ_API_KEY=your-key")
    
    if together_key:
        print(f"✅ Together API Key: {together_key[:15]}...")
    else:
        print("⚠️  Together API Key: Not configured")
        print("   Set with: export TOGETHER_API_KEY=your-key")
    
    if not groq_key and not together_key:
        print("❌ No AI APIs configured! Please set at least one API key.")
    
    print("\n📱 Android API Endpoints:")
    print("   POST /api/analyze-chart - Chart analysis")
    print("   GET  /api/test - Test AI APIs")
    print("   GET  /api/stats - Usage statistics")
    
    # Clean up old files on startup
    try:
        cleanup_old_files()
        print("✅ File cleanup completed")
    except Exception as e:
        print(f"⚠️  File cleanup failed: {e}")
    
    print("\n🚀 Starting Flask development server...")
    print("Press Ctrl+C to stop")
    
    # Start the Flask app
    app.run(debug=True, port=8081, host='0.0.0.0')
