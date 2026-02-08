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
                    
                    all_articles.append({
                        'source': source_name,
                        'title': entry.get('title', 'No title'),
                        'link': entry.get('link', '#'),
                        'summary': entry.get('summary', 'No summary available')[:300],
                        'published': entry.get('published', 'Unknown date'),
                        'date': article_date,
                        'companies': company_tags if company_tags else ['General'],
                    })
                    
            except Exception as e:
                st.warning(f"Could not fetch from {source_name}: {str(e)}")
                continue
        
        # Sort by date (most recent first)
        all_articles.sort(key=lambda x: x.get('date') or datetime.min, reverse=True)
        
        return all_articles[:max_articles]
    
    def collect_tools(self, company_filter=None):
        """Collect information about new AI tools, categorized by company"""
        
        # Comprehensive AI tools database with company affiliations
        all_tools = [
            # OpenAI
            {
                'name': 'GPT-4 Turbo',
                'company': 'OpenAI',
                'category': 'LLM',
                'description': 'Latest GPT-4 model with 128K context window and improved performance',
                'use_case': 'Advanced text generation, code assistance, analysis',
                'pricing': 'API-based',
                'link': 'https://platform.openai.com/docs/models/gpt-4-turbo'
            },
            {
                'name': 'DALL-E 3',
                'company': 'OpenAI',
                'category': 'Image Generation',
                'description': 'Advanced text-to-image generation with improved prompt following',
                'use_case': 'Creating marketing visuals, design concepts, illustrations',
                'pricing': 'API-based',
                'link': 'https://openai.com/dall-e-3'
            },
            {
                'name': 'GPT-4 Vision',
                'company': 'OpenAI',
                'category': 'Multimodal LLM',
                'description': 'GPT-4 with image understanding capabilities',
                'use_case': 'Image analysis, document processing, visual QA',
                'pricing': 'API-based',
                'link': 'https://platform.openai.com/docs/guides/vision'
            },
            
            # Google/Gemini
            {
                'name': 'Gemini Pro',
                'company': 'Google/Gemini',
                'category': 'LLM',
                'description': 'Google\'s most capable AI model with multimodal understanding',
                'use_case': 'Complex reasoning, code generation, content creation',
                'pricing': 'Free tier + API',
                'link': 'https://ai.google.dev/'
            },
            {
                'name': 'Gemini Vision',
                'company': 'Google/Gemini',
                'category': 'Multimodal LLM',
                'description': 'Gemini with native image and video understanding',
                'use_case': 'Video analysis, image captioning, visual reasoning',
                'pricing': 'API-based',
                'link': 'https://ai.google.dev/tutorials/vision_quickstart'
            },
            {
                'name': 'Vertex AI',
                'company': 'Google/Gemini',
                'category': 'ML Platform',
                'description': 'Unified AI platform for building and deploying ML models',
                'use_case': 'Enterprise AI deployment, model training, MLOps',
                'pricing': 'Pay-as-you-go',
                'link': 'https://cloud.google.com/vertex-ai'
            },
            
            # Anthropic/Claude
            {
                'name': 'Claude 3 Opus',
                'company': 'Anthropic/Claude',
                'category': 'LLM',
                'description': 'Most capable Claude model with strong reasoning and analysis',
                'use_case': 'Complex research, coding, detailed analysis',
                'pricing': 'API-based',
                'link': 'https://www.anthropic.com/claude'
            },
            {
                'name': 'Claude 3 Sonnet',
                'company': 'Anthropic/Claude',
                'category': 'LLM',
                'description': 'Balanced performance and speed for everyday tasks',
                'use_case': 'Customer service, content generation, data processing',
                'pricing': 'API-based',
                'link': 'https://www.anthropic.com/claude'
            },
            
            # NVIDIA
            {
                'name': 'NVIDIA NIM',
                'company': 'NVIDIA',
                'category': 'Inference Platform',
                'description': 'Optimized inference microservices for AI models',
                'use_case': 'Fast model deployment, edge AI, production inference',
                'pricing': 'Enterprise',
                'link': 'https://www.nvidia.com/en-us/ai-data-science/products/nim/'
            },
            {
                'name': 'NVIDIA AI Workbench',
                'company': 'NVIDIA',
                'category': 'Development Platform',
                'description': 'Unified toolkit for AI development and deployment',
                'use_case': 'Local AI development, model fine-tuning, testing',
                'pricing': 'Free',
                'link': 'https://www.nvidia.com/en-us/deep-learning-ai/solutions/data-science/workbench/'
            },
            
            # Microsoft
            {
                'name': 'Azure OpenAI Service',
                'company': 'Microsoft',
                'category': 'Cloud AI Platform',
                'description': 'Enterprise OpenAI models on Azure with security features',
                'use_case': 'Enterprise AI applications, secure deployments',
                'pricing': 'Enterprise',
                'link': 'https://azure.microsoft.com/en-us/products/ai-services/openai-service'
            },
            {
                'name': 'Copilot Studio',
                'company': 'Microsoft',
                'category': 'AI Assistant Builder',
                'description': 'Build custom copilots for Microsoft 365',
                'use_case': 'Custom AI assistants, workflow automation',
                'pricing': 'Subscription',
                'link': 'https://www.microsoft.com/en-us/microsoft-copilot/microsoft-copilot-studio'
            },
            
            # Meta/LLaMA
            {
                'name': 'LLaMA 3',
                'company': 'Meta/LLaMA',
                'category': 'Open Source LLM',
                'description': 'Meta\'s latest open-source large language model',
                'use_case': 'Research, fine-tuning, self-hosted AI',
                'pricing': 'Open Source',
                'link': 'https://llama.meta.com/'
            },
            
            # Mistral AI
            {
                'name': 'Mistral Large',
                'company': 'Mistral AI',
                'category': 'LLM',
                'description': 'High-performance multilingual AI model',
                'use_case': 'Multilingual applications, code generation',
                'pricing': 'API-based',
                'link': 'https://mistral.ai/'
            },
            
            # Cohere
            {
                'name': 'Cohere Rerank',
                'company': 'Cohere',
                'category': 'Search & Retrieval',
                'description': 'Advanced semantic reranking for search results',
                'use_case': 'Improving RAG system accuracy, search enhancement',
                'pricing': 'Freemium',
                'link': 'https://cohere.com/rerank'
            },
            
            # Stability AI
            {
                'name': 'Stable Diffusion XL',
                'company': 'Stability AI',
                'category': 'Image Generation',
                'description': 'Latest Stable Diffusion with improved image quality',
                'use_case': 'High-quality image generation, art creation',
                'pricing': 'Open Source + API',
                'link': 'https://stability.ai/stable-diffusion'
            },
            
            # Hugging Face
            {
                'name': 'Transformers',
                'company': 'Hugging Face',
                'category': 'ML Library',
                'description': 'State-of-the-art NLP library with thousands of models',
                'use_case': 'Model deployment, fine-tuning, research',
                'pricing': 'Open Source',
                'link': 'https://huggingface.co/transformers/'
            },
            {
                'name': 'Inference Endpoints',
                'company': 'Hugging Face',
                'category': 'Model Hosting',
                'description': 'Managed infrastructure for deploying ML models',
                'use_case': 'Production model serving, scalable inference',
                'pricing': 'Pay-as-you-go',
                'link': 'https://huggingface.co/inference-endpoints'
            },
            
            # General/Framework Tools
            {
                'name': 'LangChain',
                'company': 'General',
                'category': 'LLM Framework',
                'description': 'Framework for building LLM applications',
                'use_case': 'RAG systems, agents, LLM workflows',
                'pricing': 'Open Source',
                'link': 'https://www.langchain.com/'
            },
            {
                'name': 'AutoGen',
                'company': 'Microsoft',
                'category': 'Multi-Agent Framework',
                'description': 'Framework for building multi-agent AI applications',
                'use_case': 'Conversational AI, autonomous agents',
                'pricing': 'Open Source',
                'link': 'https://microsoft.github.io/autogen/'
            },
        ]
        
        # Filter by company if specified
        if company_filter and 'All Companies' not in company_filter:
            filtered_tools = [
                tool for tool in all_tools 
                if tool['company'] in company_filter or tool['company'] == 'General'
            ]
            return filtered_tools
        
        return all_tools
    
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
    
    def analyze_with_gemini(self, news_data, tools_data, research_data, custom_instructions=""):
        """Use Gemini to analyze and create intelligent report"""
        
        # Base prompt
        prompt = f"""
You are an expert AI Technology Intelligence Analyst. Analyze the following data and create a comprehensive, insightful daily intelligence report.

TODAY'S DATE: {datetime.now().strftime('%A, %B %d, %Y')}

=== NEWS ARTICLES ===
{json.dumps(news_data, indent=2)}

=== NEW AI TOOLS ===
{json.dumps(tools_data, indent=2)}

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

2. TOP AI NEWS STORIES (Select the 5 most significant stories)
   For each story provide:
   - Clear, engaging headline
   - 2-3 sentence summary focusing on WHY it matters
   - Key implications for AI practitioners
   - Source and link

3. NEW AI TOOLS SPOTLIGHT (Highlight 5 most interesting tools)
   For each tool provide:
   - Tool name and category
   - Core functionality in simple terms
   - Specific use case example
   - Why it's noteworthy or different
   - Pricing/availability

4. RESEARCH HIGHLIGHTS (If applicable)
   - Brief mention of significant papers
   - Key findings or methodologies
   - Potential real-world applications

5. EMERGING TRENDS ANALYSIS
   - Identify 2-3 patterns across today's news, tools, and research
   - What themes are emerging?
   - What technologies are gaining momentum?

6. ACTIONABLE INSIGHTS
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
            'tools': [],
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
        
        with st.spinner('🛠️ Discovering AI tools and products...'):
            report_data['tools'] = self.collect_tools(company_filter=company_filter)
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
                report_data['tools'],
                report_data['research'],
                custom_prompt
            )
        
        return report_data


def render_header():
    """Render the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Intelligence Agent</h1>
        <p>Your Daily AI Technology Intelligence Report | Powered by Gemini</p>
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
                "All Companies"
            ],
            default=["All Companies"],
            help="Filter news and tools from specific companies"
        )
        
        st.session_state.company_filter = companies
        
        st.markdown("---")
        
        auto_refresh = st.checkbox("Auto-refresh daily", value=False)
        if auto_refresh:
            st.info("Reports will auto-generate at 6 AM")
        
        include_research = st.checkbox("Include research papers", value=True)
        max_articles = st.slider("Max news articles", 5, 50, 20)
        
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
        - 📰 Latest AI news
        - 🛠️ New AI tools & frameworks
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
            <div class="metric-value">{len(report_data['tools'])}</div>
            <div class="metric-label">AI Tools</div>
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
            ### 📰 News Aggregation
            - Multiple top AI sources
            - Real-time RSS feeds
            - Smart filtering
            """)
        
        with col2:
            st.markdown("""
            ### 🛠️ Tool Discovery
            - Latest AI frameworks
            - Use case analysis
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
        "📰 Latest News",
        "🛠️ AI Tools",
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

### AI Tools
{json.dumps(report_data['tools'], indent=2)}

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
        st.markdown("## 📰 Latest AI News")
        
        if st.button("🔄 Fetch Latest News"):
            try:
                agent = AIIntelligenceAgent(st.session_state.api_key)
                date_range = st.session_state.get('date_range')
                company_filter = st.session_state.get('company_filter', ['All Companies'])
                
                news_articles = agent.collect_news(
                    max_articles=30,
                    date_range=date_range,
                    company_filter=company_filter
                )
                
                if news_articles:
                    st.success(f"✅ Found {len(news_articles)} articles")
                    
                    for article in news_articles:
                        # Create company badges
                        company_badges = ""
                        for company in article.get('companies', ['General']):
                            company_badges += f'<span class="badge" style="background: #e0e7ff; color: #667eea; margin-right: 0.25rem;">{company}</span>'
                        
                        st.markdown(f"""
                        <div class="news-card">
                            <span class="badge badge-new">NEW</span>
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
                else:
                    st.info("No articles found for the selected filters. Try adjusting your date range or company selection.")
                    
            except Exception as e:
                st.error(f"Error fetching news: {str(e)}")
    
    with tab3:
        st.markdown("## 🛠️ New AI Tools & Frameworks")
        
        if st.button("🔍 Discover AI Tools"):
            try:
                agent = AIIntelligenceAgent(st.session_state.api_key)
                company_filter = st.session_state.get('company_filter', ['All Companies'])
                tools = agent.collect_tools(company_filter=company_filter)
                
                if tools:
                    st.success(f"✅ Found {len(tools)} tools from selected companies")
                    
                    for tool in tools:
                        st.markdown(f"""
                        <div class="tool-card">
                            <h3 style="margin: 0 0 0.5rem 0;">{tool['name']}</h3>
                            <span class="badge" style="background: #ddd6fe; color: #7c3aed;">{tool['category']}</span>
                            <span class="badge" style="background: #fef3c7; color: #d97706;">{tool['pricing']}</span>
                            <span class="badge" style="background: #dbeafe; color: #2563eb;">{tool['company']}</span>
                            <p style="margin: 1rem 0; color: #334155;">{tool['description']}</p>
                            <p style="margin: 0.5rem 0;">
                                <strong>💡 Use Case:</strong> {tool['use_case']}
                            </p>
                            <a href="{tool['link']}" target="_blank" style="display: inline-block; margin-top: 0.75rem;">
                                Explore tool →
                            </a>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No tools found for the selected companies.")
                    
            except Exception as e:
                st.error(f"Error fetching tools: {str(e)}")
    
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
