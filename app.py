import streamlit as st
import google.generativeai as genai
from datetime import datetime, timedelta
import feedparser
import requests
from bs4 import BeautifulSoup
import json
import time
import pandas as pd
from urllib.parse import quote_plus
import os

# Page Configuration
st.set_page_config(
    page_title="AI Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --background-color: #0f172a;
        --card-background: #1e293b;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: #e0e7ff;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .news-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .tool-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    /* Category badges */
    .category-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    
    .badge-new {
        background: #10b981;
        color: white;
    }
    
    .badge-trending {
        background: #f59e0b;
        color: white;
    }
    
    .badge-research {
        background: #3b82f6;
        color: white;
    }
    
    .badge-lawsuit {
        background: #ef4444;
        color: white;
    }
    
    .badge-technology {
        background: #8b5cf6;
        color: white;
    }
    
    .badge-hardware {
        background: #06b6d4;
        color: white;
    }
    
    .badge-software {
        background: #10b981;
        color: white;
    }
    
    .badge-stocks {
        background: #f59e0b;
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #64748b;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8fafc;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
    
    /* Loading animation */
    .loading-animation {
        text-align: center;
        padding: 2rem;
    }
    
    /* Status indicators */
    .status-online {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
    }
    
    /* Links */
    a {
        color: #667eea;
        text-decoration: none;
    }
    
    a:hover {
        color: #764ba2;
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'reports_history' not in st.session_state:
    st.session_state.reports_history = []
if 'api_key' not in st.session_state:
    # Try to get API key from Streamlit secrets first
    try:
        st.session_state.api_key = st.secrets.get("GEMINI_API_KEY", None)
    except:
        st.session_state.api_key = None
if 'last_generated' not in st.session_state:
    st.session_state.last_generated = None

class AIIntelligenceAgent:
    """Main AI Intelligence Agent class"""
    
    def __init__(self, api_key):
        """Initialize the agent with Gemini API"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        self.news_sources = {
            'TechCrunch AI': 'https://techcrunch.com/category/artificial-intelligence/feed/',
            'MIT Technology Review': 'https://www.technologyreview.com/feed/',
            'VentureBeat AI': 'https://venturebeat.com/category/ai/feed/',
            'OpenAI Blog': 'https://openai.com/blog/rss/',
            'The Batch (DeepLearning.AI)': 'https://www.deeplearning.ai/the-batch/feed/',
        }
    
    def categorize_article(self, article):
        """Categorize article based on content"""
        title = article.get('title', '').lower()
        summary = article.get('summary', '').lower()
        content = title + ' ' + summary
        
        categories = []
        
        # Lawsuit detection
        lawsuit_keywords = ['lawsuit', 'sue', 'suing', 'legal action', 'court', 'patent infringement', 
                           'litigation', 'settlement', 'copyright', 'intellectual property']
        if any(keyword in content for keyword in lawsuit_keywords):
            categories.append('Lawsuits')
        
        # Hardware detection
        hardware_keywords = ['chip', 'gpu', 'processor', 'hardware', 'nvidia', 'tpu', 'semiconductor',
                           'h100', 'a100', 'data center', 'server', 'silicon']
        if any(keyword in content for keyword in hardware_keywords):
            categories.append('Hardware')
        
        # Software/Tools detection
        software_keywords = ['api', 'sdk', 'framework', 'library', 'platform', 'application',
                           'software', 'tool', 'plugin', 'integration', 'update', 'release']
        if any(keyword in content for keyword in software_keywords):
            categories.append('Software & Tools')
        
        # Stocks/Finance detection
        stocks_keywords = ['stock', 'share', 'investment', 'funding', 'valuation', 'ipo', 
                          'market cap', 'revenue', 'earnings', 'investor', 'venture capital']
        if any(keyword in content for keyword in stocks_keywords):
            categories.append('Stocks & Finance')
        
        # Technology/Research detection
        tech_keywords = ['model', 'algorithm', 'research', 'paper', 'breakthrough', 'technique',
                        'architecture', 'training', 'inference', 'benchmark', 'performance']
        if any(keyword in content for keyword in tech_keywords):
            categories.append('Technology & Research')
        
        # If no category matched, mark as General
        if not categories:
            categories.append('General')
        
        return categories
    
    def collect_news(self, max_articles=20, date_range=None, company_filter=None):
        """Collect latest AI news from RSS feeds with date and company filtering"""
        
        # Enhanced news sources with company-specific feeds
        all_news_sources = {
            'General': {
                'TechCrunch AI': 'https://techcrunch.com/category/artificial-intelligence/feed/',
                'MIT Technology Review': 'https://www.technologyreview.com/feed/',
                'VentureBeat AI': 'https://venturebeat.com/category/ai/feed/',
                'The Batch (DeepLearning.AI)': 'https://www.deeplearning.ai/the-batch/feed/',
                'AI News': 'https://artificialintelligence-news.com/feed/',
            },
            'OpenAI': {
                'OpenAI Blog': 'https://openai.com/blog/rss/',
            },
            'Google/Gemini': {
                'Google AI Blog': 'https://blog.google/technology/ai/rss/',
                'Google DeepMind': 'https://deepmind.google/blog/rss.xml',
            },
            'Anthropic/Claude': {
                'Anthropic News': 'https://www.anthropic.com/news/rss',
            },
            'NVIDIA': {
                'NVIDIA Blog': 'https://blogs.nvidia.com/feed/',
            },
            'Microsoft': {
                'Microsoft AI Blog': 'https://blogs.microsoft.com/ai/feed/',
            },
            'Meta/LLaMA': {
                'Meta AI Blog': 'https://ai.meta.com/blog/rss/',
            },
            'Hugging Face': {
                'Hugging Face Blog': 'https://huggingface.co/blog/feed.xml',
            },
        }
        
        # Determine which sources to use based on company filter
        sources_to_use = {}
        
        if not company_filter or 'All Companies' in company_filter:
            # Use all sources
            for category, sources in all_news_sources.items():
                sources_to_use.update(sources)
        else:
            # Use general sources always
            sources_to_use.update(all_news_sources.get('General', {}))
            
            # Add specific company sources
            for company in company_filter:
                if company in all_news_sources:
                    sources_to_use.update(all_news_sources[company])
        
        all_articles = []
        
        # Parse date range if provided
        start_date = None
        end_date = None
        if date_range:
            start_date, end_date = date_range
            # Convert to datetime for comparison
            start_date = datetime.combine(start_date, datetime.min.time())
            end_date = datetime.combine(end_date, datetime.max.time())
        
        for source_name, feed_url in sources_to_use.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    # Parse published date
                    article_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        article_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        article_date = datetime(*entry.updated_parsed[:6])
                    
                    # Filter by date range if specified
                    if date_range and article_date:
                        if not (start_date <= article_date <= end_date):
                            continue
                    
                    # Determine company tags
                    company_tags = []
                    title_lower = entry.get('title', '').lower()
                    summary_lower = entry.get('summary', '').lower()
                    
                    if 'openai' in title_lower or 'gpt' in title_lower or 'chatgpt' in title_lower:
                        company_tags.append('OpenAI')
                    if 'gemini' in title_lower or 'google' in title_lower or 'deepmind' in title_lower:
                        company_tags.append('Google/Gemini')
                    if 'claude' in title_lower or 'anthropic' in title_lower:
                        company_tags.append('Anthropic/Claude')
                    if 'nvidia' in title_lower:
                        company_tags.append('NVIDIA')
                    if 'microsoft' in title_lower:
                        company_tags.append('Microsoft')
                    if 'meta' in title_lower or 'llama' in title_lower or 'facebook' in title_lower:
                        company_tags.append('Meta/LLaMA')
                    if 'mistral' in title_lower:
                        company_tags.append('Mistral AI')
                    if 'cohere' in title_lower:
                        company_tags.append('Cohere')
                    if 'stability' in title_lower:
                        company_tags.append('Stability AI')
                    if 'hugging face' in title_lower or 'huggingface' in title_lower:
                        company_tags.append('Hugging Face')
                    
                    article = {
                        'source': source_name,
                        'title': entry.get('title', 'No title'),
                        'link': entry.get('link', '#'),
                        'summary': entry.get('summary', 'No summary available')[:300],
                        'published': entry.get('published', 'Unknown date'),
                        'date': article_date,
                        'companies': company_tags if company_tags else ['General'],
                    }
                    
                    # Categorize the article
                    article['categories'] = self.categorize_article(article)
                    
                    all_articles.append(article)
                    
            except Exception as e:
                st.warning(f"Could not fetch from {source_name}: {str(e)}")
                continue
        
        # Sort by date (most recent first)
        all_articles.sort(key=lambda x: x.get('date') or datetime.min, reverse=True)
        
        return all_articles[:max_articles]
    
    def collect_latest_models(self, company_filter=None):
        """Collect information about latest AI model developments and releases"""
        
        # Latest AI models and developments (Updated regularly)
        latest_models = [
            # OpenAI - Latest Models
            {
                'name': 'GPT-4o (GPT-4 Omni)',
                'company': 'OpenAI',
                'release_date': 'May 2024',
                'category': 'Multimodal LLM',
                'description': 'Flagship multimodal model with vision, audio, and text capabilities',
                'key_features': ['128K context', 'Native multimodal', 'Faster than GPT-4 Turbo', 'Better at vision and audio'],
                'benchmarks': 'MMLU: 88.7%, HumanEval: 90.2%',
                'pricing': 'API-based: $5/1M input, $15/1M output tokens',
                'status': 'Production',
                'link': 'https://openai.com/index/hello-gpt-4o/'
            },
            {
                'name': 'o1 & o1-mini',
                'company': 'OpenAI',
                'release_date': 'September 2024',
                'category': 'Reasoning LLM',
                'description': 'Advanced reasoning models that think before they answer',
                'key_features': ['Chain-of-thought reasoning', 'PhD-level science', 'Complex problem solving', 'Math & coding excellence'],
                'benchmarks': 'AIME 2024: 83% (vs 13% GPT-4o), Codeforces: 89th percentile',
                'pricing': 'o1: $15/1M in, $60/1M out | o1-mini: $3/1M in, $12/1M out',
                'status': 'Production',
                'link': 'https://openai.com/o1/'
            },
            
            # Google/Gemini - Latest Models
            {
                'name': 'Gemini 2.0 Flash',
                'company': 'Google/Gemini',
                'release_date': 'December 2024',
                'category': 'Multimodal LLM',
                'description': 'Next-generation thinking model with agentic capabilities',
                'key_features': ['Multimodal input/output', 'Native tool use', 'Agentic behavior', '2x faster than 1.5'],
                'benchmarks': 'MMLU-Pro: 75.1%, Math-500: 71.9%, HumanEval: 92.7%',
                'pricing': 'Free tier available, API pricing competitive',
                'status': 'Production',
                'link': 'https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/'
            },
            {
                'name': 'Gemini 1.5 Pro',
                'company': 'Google/Gemini',
                'release_date': 'February 2024',
                'category': 'Long-Context LLM',
                'description': 'Extended context window up to 2M tokens with multimodal understanding',
                'key_features': ['2M token context', 'Video understanding', 'Code execution', 'Function calling'],
                'benchmarks': 'MMLU: 85.9%, Long-context recall: 99.7%',
                'pricing': 'Pay-as-you-go with free tier',
                'status': 'Production',
                'link': 'https://deepmind.google/technologies/gemini/pro/'
            },
            
            # Anthropic/Claude - Latest Models
            {
                'name': 'Claude 3.5 Sonnet',
                'company': 'Anthropic/Claude',
                'release_date': 'October 2024',
                'category': 'LLM',
                'description': 'Most intelligent Claude model with enhanced coding and agentic capabilities',
                'key_features': ['Computer use (beta)', 'Superior coding', '200K context', 'Vision capabilities'],
                'benchmarks': 'SWE-bench Verified: 49.0%, TAU-bench: 69.2%, GPQA: 65.0%',
                'pricing': '$3/1M input, $15/1M output tokens',
                'status': 'Production',
                'link': 'https://www.anthropic.com/claude/sonnet'
            },
            {
                'name': 'Claude 3.5 Haiku',
                'company': 'Anthropic/Claude',
                'release_date': 'November 2024',
                'category': 'Fast LLM',
                'description': 'Fastest and most affordable Claude 3.5 model',
                'key_features': ['Low latency', 'Vision support', 'Same quality as Opus 3', 'Cost-effective'],
                'benchmarks': 'Coding: rivals Claude 3 Opus, Speed: 3x faster',
                'pricing': '$0.80/1M input, $4/1M output tokens',
                'status': 'Production',
                'link': 'https://www.anthropic.com/claude/haiku'
            },
            
            # Meta/LLaMA - Latest Models
            {
                'name': 'Llama 3.3 70B',
                'company': 'Meta/LLaMA',
                'release_date': 'December 2024',
                'category': 'Open Source LLM',
                'description': 'Cost-effective open model matching 405B performance',
                'key_features': ['70B parameters', '405B-level performance', 'Multilingual', 'Commercial license'],
                'benchmarks': 'MMLU: 86.0%, HumanEval: 88.4%, MATH: 71.7%',
                'pricing': 'Open source (free)',
                'status': 'Production',
                'link': 'https://ai.meta.com/blog/llama-3-3/'
            },
            {
                'name': 'Llama 3.2 Vision',
                'company': 'Meta/LLaMA',
                'release_date': 'September 2024',
                'category': 'Multimodal Open LLM',
                'description': 'First Llama models with native vision understanding',
                'key_features': ['11B & 90B sizes', 'Image understanding', 'Edge deployment', 'Open source'],
                'benchmarks': '90B: competitive with closed models on vision tasks',
                'pricing': 'Open source (free)',
                'status': 'Production',
                'link': 'https://ai.meta.com/blog/llama-3-2-vision/'
            },
            
            # Mistral AI - Latest Models
            {
                'name': 'Mistral Large 2',
                'company': 'Mistral AI',
                'release_date': 'July 2024',
                'category': 'LLM',
                'description': '123B parameter flagship model with advanced reasoning',
                'key_features': ['123B parameters', '128K context', 'Function calling', 'Multilingual (80+ languages)'],
                'benchmarks': 'MMLU: 84.0%, HumanEval: 92.0%, Math: 76.9%',
                'pricing': 'API-based: competitive pricing',
                'status': 'Production',
                'link': 'https://mistral.ai/news/mistral-large-2407/'
            },
            
            # DeepSeek - Latest Models
            {
                'name': 'DeepSeek-V3',
                'company': 'DeepSeek',
                'release_date': 'December 2024',
                'category': 'Open Source LLM',
                'description': '671B MoE model with groundbreaking efficiency',
                'key_features': ['671B total, 37B active', 'MoE architecture', 'Open source', 'Cost-effective training'],
                'benchmarks': 'Matches GPT-4o, trained for $5.5M',
                'pricing': 'Open source (free)',
                'status': 'Production',
                'link': 'https://github.com/deepseek-ai/DeepSeek-V3'
            },
            
            # Specialized Models
            {
                'name': 'DALL-E 3',
                'company': 'OpenAI',
                'release_date': 'October 2023 (Updated)',
                'category': 'Image Generation',
                'description': 'State-of-the-art text-to-image generation',
                'key_features': ['Better prompt following', 'Higher quality', 'Integrated ChatGPT', 'Safe generation'],
                'benchmarks': 'Human preference: 71.2% vs 48.8% (DALL-E 2)',
                'pricing': 'API-based: from $0.040/image',
                'status': 'Production',
                'link': 'https://openai.com/dall-e-3'
            },
            {
                'name': 'Stable Diffusion 3.5',
                'company': 'Stability AI',
                'release_date': 'October 2024',
                'category': 'Image Generation',
                'description': 'Latest open-source image generation with improved quality',
                'key_features': ['Multiple sizes (Large, Medium)', 'Commercial license', 'Better prompt adherence', 'Fine-tunable'],
                'benchmarks': 'Competitive with Midjourney v6 & DALL-E 3',
                'pricing': 'Open source + API options',
                'status': 'Production',
                'link': 'https://stability.ai/news/stable-diffusion-3-5'
            },
            {
                'name': 'Whisper v3',
                'company': 'OpenAI',
                'release_date': 'November 2023 (Updated)',
                'category': 'Speech Recognition',
                'description': 'Multilingual speech recognition and translation',
                'key_features': ['99 languages', 'Real-time capable', 'Robust to accents', 'Open source'],
                'benchmarks': 'WER improvements across all languages',
                'pricing': 'Open source (free) + API',
                'status': 'Production',
                'link': 'https://github.com/openai/whisper'
            },
        ]
        
        # Filter by company if specified
        if company_filter and 'All Companies' not in company_filter:
            filtered_models = [
                model for model in latest_models 
                if model['company'] in company_filter
            ]
            return filtered_models
        
        return latest_models
    
    def collect_research(self):
        """Collect recent AI research papers"""
        # Mock data - in production, integrate with arXiv API
        papers = [
            {
                'title': 'Constitutional AI: Harmlessness from AI Feedback',
                'authors': 'Anthropic Research Team',
                'summary': 'A method for training AI systems to be helpful, harmless, and honest using AI feedback',
                'category': 'AI Safety',
                'link': 'https://arxiv.org/abs/2212.08073'
            },
            {
                'title': 'Retrieval-Augmented Generation for Knowledge-Intensive Tasks',
                'authors': 'Facebook AI Research',
                'summary': 'Combining retrieval with generation for improved factual accuracy',
                'category': 'RAG Systems',
                'link': 'https://arxiv.org/abs/2005.11401'
            }
        ]
        return papers
    
    def analyze_with_gemini(self, news_data, models_data, research_data, custom_instructions=""):
        """Use Gemini to analyze and create intelligent report"""
        
        # Base prompt
        prompt = f"""
You are an expert AI Technology Intelligence Analyst. Analyze the following data and create a comprehensive, insightful daily intelligence report.

TODAY'S DATE: {datetime.now().strftime('%A, %B %d, %Y')}

=== NEWS ARTICLES (BY CATEGORY) ===
{json.dumps(news_data, indent=2)}

=== LATEST AI MODEL DEVELOPMENTS ===
{json.dumps(models_data, indent=2)}

=== RESEARCH PAPERS ===
{json.dumps(research_data, indent=2)}
"""
        
        # Add custom instructions if provided
        if custom_instructions and custom_instructions.strip():
            prompt += f"""

=== CUSTOM FOCUS INSTRUCTIONS ===
{custom_instructions}

IMPORTANT: Pay special attention to the custom instructions above when analyzing and formatting your report.
"""
        
        prompt += """

Create a professional intelligence report with the following sections:

1. EXECUTIVE SUMMARY (2-3 sentences highlighting the most important developments)

2. NEWS BY CATEGORY
   Organize news into these categories and highlight top stories in each:
   - 🏛️ LAWSUITS & LEGAL: Any legal battles, patent disputes, regulatory issues
   - 💻 TECHNOLOGY & RESEARCH: New techniques, algorithms, breakthroughs
   - 🔧 HARDWARE: Chips, GPUs, infrastructure developments
   - 🛠️ SOFTWARE & TOOLS: New releases, updates, features
   - 📈 STOCKS & FINANCE: Funding, valuations, market movements
   - 📰 GENERAL: Other significant news
   
   For each category (only include if there's news):
   - Section header with emoji
   - Top 2-3 most significant stories
   - Brief analysis of implications

3. LATEST MODEL DEVELOPMENTS
   Highlight the newest and most significant AI models:
   - Model name, company, and release date
   - Key capabilities and features
   - Benchmark performance highlights
   - Why this matters / Use cases
   - Focus on models released or updated in the last 3-6 months

4. EMERGING TRENDS ANALYSIS
   - Identify 2-3 patterns across today's news, models, and developments
   - What themes are emerging?
   - Which companies are leading in which areas?

5. ACTIONABLE INSIGHTS
   - 3 specific takeaways for AI developers/enthusiasts
   - Concrete actions or areas to explore

Format your response in clean, professional markdown. Use headers, bullet points, and bold text appropriately.
Be specific, insightful, and avoid generic statements. Focus on what's genuinely useful and interesting.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    def generate_daily_report(self):
        """Main orchestration method"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d'),
            'news': [],
            'models': [],
            'research': [],
            'analysis': '',
            'filters': {
                'date_range': st.session_state.get('date_range'),
                'companies': st.session_state.get('company_filter', ['All Companies'])
            }
        }
        
        # Get filters from session state
        date_range = st.session_state.get('date_range')
        company_filter = st.session_state.get('company_filter', ['All Companies'])
        
        # Collect data with filters
        with st.spinner('🔍 Collecting AI news from selected companies...'):
            report_data['news'] = self.collect_news(
                max_articles=50,
                date_range=date_range,
                company_filter=company_filter
            )
            time.sleep(1)
        
        with st.spinner('🤖 Discovering latest AI models and developments...'):
            report_data['models'] = self.collect_latest_models(company_filter=company_filter)
            time.sleep(1)
        
        with st.spinner('🔬 Gathering research papers...'):
            report_data['research'] = self.collect_research()
            time.sleep(1)
        
        # Get custom prompt from session state
        custom_prompt = st.session_state.get('custom_prompt', '')
        
        # Analyze with Gemini
        analysis_message = '🤖 Analyzing data with Gemini AI...'
        if custom_prompt:
            analysis_message = '🤖 Analyzing with your custom focus...'
        
        with st.spinner(analysis_message):
            report_data['analysis'] = self.analyze_with_gemini(
                report_data['news'],
                report_data['models'],
                report_data['research'],
                custom_prompt
            )
        
        return report_data


def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖  Prama AI</h1>
        <p>Your Daily AI News Technology updates & Intelligence Report</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with configuration"""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # Check if API key is already loaded from secrets
        api_key_from_secrets = False
        try:
            if st.secrets.get("GEMINI_API_KEY"):
                api_key_from_secrets = True
                st.success("✅ API Key loaded from secrets")
        except:
            pass
        
        # API Key input (optional if already in secrets)
        if not api_key_from_secrets:
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.api_key or "",
                help="Enter your Google Gemini API key"
            )
            
            if api_key:
                st.session_state.api_key = api_key
                st.success("✅ API Key configured")
            else:
                st.warning("⚠️ Please enter your Gemini API key")
                st.markdown("""
                [Get your API key](https://makersuite.google.com/app/apikey) from Google AI Studio
                """)
        else:
            # Show option to override (collapsed by default)
            with st.expander("🔧 Use Different API Key"):
                override_key = st.text_input(
                    "Enter alternative API key",
                    type="password",
                    help="Override the API key from secrets if needed"
                )
                if override_key:
                    st.session_state.api_key = override_key
                    st.success("✅ Using override API key")
        
        st.markdown("---")
        
        # Report settings
        st.markdown("### 📊 Report Settings")
        
        # Date Range Selector
        st.markdown("#### 📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "From",
                value=datetime.now() - timedelta(days=7),
                max_value=datetime.now(),
                help="Start date for news collection"
            )
        with col2:
            end_date = st.date_input(
                "To",
                value=datetime.now(),
                max_value=datetime.now(),
                help="End date for news collection"
            )
        
        # Store in session state
        st.session_state.date_range = (start_date, end_date)
        
        # Calculate days
        days_diff = (end_date - start_date).days
        if days_diff > 0:
            st.caption(f"📊 Analyzing {days_diff} days of AI developments")
        
        st.markdown("---")
        
        # Company/Source Filter
        st.markdown("#### 🏢 Focus on Companies")
        
        companies = st.multiselect(
            "Select AI Companies to Track",
            options=[
                "OpenAI",
                "Google/Gemini", 
                "Anthropic/Claude",
                "NVIDIA",
                "Microsoft",
                "Meta/LLaMA",
                "Mistral AI",
                "Cohere",
                "Stability AI",
                "Hugging Face",
                "DeepSeek",
                "All Companies"
            ],
            default=["All Companies"],
            help="Filter news and models from specific companies"
        )
        
        st.session_state.company_filter = companies
        
        st.markdown("---")
        
        # Custom Prompt Section
        st.markdown("### 💬 Custom Prompt")
        
        with st.expander("🎯 Customize Analysis Focus"):
            st.markdown("""
            Add custom instructions to focus the AI analysis on specific topics, 
            industries, or perspectives that matter to you.
            """)
            
            custom_prompt = st.text_area(
                "Additional Instructions",
                height=150,
                placeholder="""Examples:
- Focus on healthcare AI applications
- Emphasize business implications over technical details
- Highlight startup opportunities
- Compare with competitors in the fintech space
- Include security and privacy considerations""",
                help="These instructions will be added to the analysis prompt"
            )
            
            # Save to session state
            if 'custom_prompt' not in st.session_state:
                st.session_state.custom_prompt = ""
            
            if custom_prompt != st.session_state.custom_prompt:
                st.session_state.custom_prompt = custom_prompt
                if custom_prompt:
                    st.success("✅ Custom instructions saved")
            
            # Quick templates
            st.markdown("**Quick Templates:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🏥 Healthcare Focus", use_container_width=True):
                    st.session_state.custom_prompt = "Focus on healthcare and medical AI applications. Highlight clinical use cases, FDA/regulatory considerations, and patient impact."
                    st.rerun()
                
                if st.button("💼 Business/Enterprise", use_container_width=True):
                    st.session_state.custom_prompt = "Emphasize business value, ROI, enterprise adoption, and practical implementation strategies. Focus on B2B opportunities."
                    st.rerun()
            
            with col2:
                if st.button("🔐 Security & Privacy", use_container_width=True):
                    st.session_state.custom_prompt = "Highlight security implications, privacy concerns, data protection, and ethical considerations in AI developments."
                    st.rerun()
                
                if st.button("🚀 Startups & Innovation", use_container_width=True):
                    st.session_state.custom_prompt = "Focus on startup opportunities, innovation trends, funding news, and emerging market gaps."
                    st.rerun()
            
            if st.session_state.custom_prompt:
                if st.button("🗑️ Clear Custom Prompt", use_container_width=True):
                    st.session_state.custom_prompt = ""
                    st.rerun()
        
        st.markdown("---")
        
        # Statistics
        st.markdown("### 📈 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Reports Generated", len(st.session_state.reports_history))
        with col2:
            if st.session_state.last_generated:
                hours_ago = int((datetime.now() - st.session_state.last_generated).total_seconds() / 3600)
                st.metric("Last Report", f"{hours_ago}h ago")
        
        st.markdown("---")
        
        # About
        st.markdown("### ℹ️ About")
        st.markdown("""
        This AI Intelligence Agent automatically collects and analyzes:
        - 📰 Latest AI news (categorized)
        - 🤖 Latest model developments
        - 🔬 Research papers
        - 📈 Emerging trends
        
        **Powered by:**
        - Google Gemini AI
        - Streamlit
        - Python
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #64748b; font-size: 0.875rem;'>
            Made with ❤️ for AI enthusiasts
        </div>
        """, unsafe_allow_html=True)


def render_metrics_dashboard(report_data):
    """Render key metrics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(report_data['news'])}</div>
            <div class="metric-label">News Articles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(report_data['models'])}</div>
            <div class="metric-label">Latest Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(report_data['research'])}</div>
            <div class="metric-label">Research Papers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">✓</div>
            <div class="metric-label">Analysis Complete</div>
        </div>
        """, unsafe_allow_html=True)


def render_news_by_category(news_articles):
    """Render news articles organized by category"""
    # Group articles by category
    categorized_news = {
        'Lawsuits': [],
        'Technology & Research': [],
        'Hardware': [],
        'Software & Tools': [],
        'Stocks & Finance': [],
        'General': []
    }
    
    for article in news_articles:
        for category in article.get('categories', ['General']):
            if category in categorized_news:
                categorized_news[category].append(article)
    
    # Category display configuration
    category_config = {
        'Lawsuits': {'emoji': '🏛️', 'badge_class': 'badge-lawsuit'},
        'Technology & Research': {'emoji': '💻', 'badge_class': 'badge-technology'},
        'Hardware': {'emoji': '🔧', 'badge_class': 'badge-hardware'},
        'Software & Tools': {'emoji': '🛠️', 'badge_class': 'badge-software'},
        'Stocks & Finance': {'emoji': '📈', 'badge_class': 'badge-stocks'},
        'General': {'emoji': '📰', 'badge_class': 'badge-new'}
    }
    
    # Display each category
    for category, articles in categorized_news.items():
        if articles:  # Only show categories with articles
            config = category_config.get(category, {'emoji': '📰', 'badge_class': 'badge-new'})
            
            st.markdown(f"""
            <div class="category-header">
                {config['emoji']} {category}
            </div>
            """, unsafe_allow_html=True)
            
            for article in articles[:5]:  # Limit to 5 articles per category
                # Create company badges
                company_badges = ""
                for company in article.get('companies', ['General']):
                    company_badges += f'<span class="badge" style="background: #e0e7ff; color: #667eea; margin-right: 0.25rem;">{company}</span>'
                
                st.markdown(f"""
                <div class="news-card">
                    <span class="badge {config['badge_class']}">{category}</span>
                    <span class="badge" style="background: #f0fdf4; color: #16a34a;">{article['source']}</span>
                    {company_badges}
                    <h3 style="margin: 0.75rem 0;">{article['title']}</h3>
                    <p style="color: #64748b; margin: 0.5rem 0;">{article['summary']}</p>
                    <div style="margin-top: 1rem;">
                        <a href="{article['link']}" target="_blank">Read more →</a>
                        <span style="color: #94a3b8; margin-left: 1rem; font-size: 0.875rem;">
                            📅 {article['published']}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def main():
    """Main application function"""
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Check if API key is configured
    if not st.session_state.api_key:
        st.warning("👈 Please configure your Gemini API key in the sidebar to get started")
        
        # Show feature overview
        st.markdown("## 🌟 Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📰 Categorized News
            - Lawsuits & Legal
            - Technology & Research
            - Hardware Developments
            - Software & Tools
            - Stocks & Finance
            """)
        
        with col2:
            st.markdown("""
            ### 🤖 Latest Models
            - New model releases
            - Performance benchmarks
            - Feature comparisons
            - Pricing information
            """)
        
        with col3:
            st.markdown("""
            ### 🤖 AI Analysis
            - Gemini-powered insights
            - Trend identification
            - Actionable recommendations
            """)
        
        return
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Generate Report",
        "📰 News by Category",
        "🤖 Latest Models",
        "📚 History"
    ])
    
    with tab1:
        st.markdown("## 🤖 AI Intelligence Report Generator")
        st.markdown("Generate a comprehensive daily intelligence report powered by Gemini AI")
        
        # Show active filters
        date_range = st.session_state.get('date_range')
        company_filter = st.session_state.get('company_filter', ['All Companies'])
        
        if date_range:
            start_date, end_date = date_range
            days = (end_date - start_date).days
            st.info(f"📅 **Date Range:** {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')} ({days} days)")
        
        if company_filter and 'All Companies' not in company_filter:
            companies_str = ", ".join(company_filter)
            st.info(f"🏢 **Tracking Companies:** {companies_str}")
        
        # Show custom prompt indicator if active
        if st.session_state.get('custom_prompt'):
            st.info(f"🎯 **Custom Focus Active:** {st.session_state.custom_prompt[:100]}{'...' if len(st.session_state.custom_prompt) > 100 else ''}")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Generate Daily Report", use_container_width=True, type="primary"):
                try:
                    agent = AIIntelligenceAgent(st.session_state.api_key)
                    report_data = agent.generate_daily_report()
                    
                    # Save to history
                    st.session_state.reports_history.insert(0, report_data)
                    st.session_state.last_generated = datetime.now()
                    
                    # Display metrics
                    st.markdown("---")
                    render_metrics_dashboard(report_data)
                    
                    # Display the analysis
                    st.markdown("---")
                    st.markdown("## 📋 Intelligence Report")
                    st.markdown(report_data['analysis'])
                    
                    # Download button
                    st.markdown("---")
                    report_text = f"""# AI Intelligence Report
Generated: {report_data['timestamp']}

{report_data['analysis']}

---
## Raw Data

### News Articles
{json.dumps(report_data['news'], indent=2)}

### Latest AI Models
{json.dumps(report_data['models'], indent=2)}

### Research Papers
{json.dumps(report_data['research'], indent=2)}
"""
                    st.download_button(
                        label="📥 Download Report",
                        data=report_text,
                        file_name=f"ai_intelligence_report_{report_data['date']}.md",
                        mime="text/markdown"
                    )
                    
                    st.success("✅ Report generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")
        
        # Show last report if available
        if st.session_state.reports_history:
            st.markdown("---")
            st.markdown("### 📄 Latest Report Preview")
            latest_report = st.session_state.reports_history[0]
            with st.expander("View latest report", expanded=False):
                st.markdown(latest_report['analysis'])
    
    with tab2:
        st.markdown("## 📰 Latest AI News by Category")
        
        if st.button("🔄 Fetch Latest News"):
            try:
                agent = AIIntelligenceAgent(st.session_state.api_key)
                date_range = st.session_state.get('date_range')
                company_filter = st.session_state.get('company_filter', ['All Companies'])
                
                news_articles = agent.collect_news(
                    max_articles=50,
                    date_range=date_range,
                    company_filter=company_filter
                )
                
                if news_articles:
                    st.success(f"✅ Found {len(news_articles)} articles")
                    render_news_by_category(news_articles)
                else:
                    st.info("No articles found for the selected filters. Try adjusting your date range or company selection.")
                    
            except Exception as e:
                st.error(f"Error fetching news: {str(e)}")
    
    with tab3:
        st.markdown("## 🤖 Latest AI Model Developments")
        st.markdown("*Discover the newest and most significant AI models released in recent months*")
        
        if st.button("🔍 Discover Latest Models"):
            try:
                agent = AIIntelligenceAgent(st.session_state.api_key)
                company_filter = st.session_state.get('company_filter', ['All Companies'])
                models = agent.collect_latest_models(company_filter=company_filter)
                
                if models:
                    st.success(f"✅ Found {len(models)} latest models from selected companies")
                    
                    for model in models:
                        # Status badge color
                        status_color = "#10b981" if model['status'] == "Production" else "#f59e0b"
                        
                        st.markdown(f"""
                        <div class="tool-card">
                            <h3 style="margin: 0 0 0.5rem 0;">{model['name']}</h3>
                            <span class="badge" style="background: #ddd6fe; color: #7c3aed;">{model['category']}</span>
                            <span class="badge" style="background: {status_color}; color: white;">{model['status']}</span>
                            <span class="badge" style="background: #dbeafe; color: #2563eb;">{model['company']}</span>
                            <span class="badge" style="background: #fef3c7; color: #d97706;">Released: {model['release_date']}</span>
                            
                            <p style="margin: 1rem 0; color: #334155; font-weight: 500;">{model['description']}</p>
                            
                            <div style="margin: 1rem 0;">
                                <strong>🎯 Key Features:</strong>
                                <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                                    {"".join([f"<li>{feature}</li>" for feature in model['key_features']])}
                                </ul>
                            </div>
                            
                            <p style="margin: 0.5rem 0;">
                                <strong>📊 Benchmarks:</strong> {model['benchmarks']}
                            </p>
                            
                            <p style="margin: 0.5rem 0;">
                                <strong>💰 Pricing:</strong> {model['pricing']}
                            </p>
                            
                            <a href="{model['link']}" target="_blank" style="display: inline-block; margin-top: 0.75rem;">
                                Learn more about {model['name']} →
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No models found for the selected companies.")
                    
            except Exception as e:
                st.error(f"Error fetching models: {str(e)}")
    
    with tab4:
        st.markdown("## 📚 Report History")
        
        if not st.session_state.reports_history:
            st.info("No reports generated yet. Generate your first report in the 'Generate Report' tab!")
        else:
            st.markdown(f"**Total Reports:** {len(st.session_state.reports_history)}")
            
            for idx, report in enumerate(st.session_state.reports_history):
                with st.expander(
                    f"📄 Report from {report['date']} ({report['timestamp'].split('T')[1][:8]})",
                    expanded=(idx == 0)
                ):
                    st.markdown(report['analysis'])
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"Generated: {report['timestamp']}")
                    with col2:
                        report_text = f"""# AI Intelligence Report
{report['analysis']}
"""
                        st.download_button(
                            label="📥 Download",
                            data=report_text,
                            file_name=f"report_{report['date']}.md",
                            mime="text/markdown",
                            key=f"download_{idx}"
                        )


if __name__ == "__main__":
    main()
