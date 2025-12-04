"""
Usage Management Module for AI Investor SaaS
Handles free tier usage tracking (5 free analyses)
"""

import streamlit as st
from datetime import datetime

# Constants
FREE_USAGE_LIMIT = 5

def initialize_usage():
    """Initialize usage tracking in session state"""
    if 'usage_count' not in st.session_state:
        st.session_state.usage_count = 0
    if 'usage_history' not in st.session_state:
        st.session_state.usage_history = []

def get_usage_count():
    """Get current usage count"""
    return st.session_state.get('usage_count', 0)

def get_remaining_analyses():
    """Get number of remaining free analyses"""
    return max(0, FREE_USAGE_LIMIT - get_usage_count())

def increment_usage(analysis_type="single"):
    """
    Increment usage counter
    Args:
        analysis_type: 'single' or 'portfolio'
    """
    if 'usage_count' not in st.session_state:
        initialize_usage()
    
    st.session_state.usage_count += 1
    st.session_state.usage_history.append({
        'timestamp': datetime.now().isoformat(),
        'type': analysis_type,
        'count': st.session_state.usage_count
    })

def can_perform_analysis():
    """Check if user can perform another analysis"""
    return get_usage_count() < FREE_USAGE_LIMIT

def reset_usage():
    """Reset usage (for testing/admin purposes)"""
    st.session_state.usage_count = 0
    st.session_state.usage_history = []

def display_usage_widget():
    """Display usage tracking widget in sidebar"""
    initialize_usage()
    
    used = get_usage_count()
    remaining = get_remaining_analyses()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Usage Tracker")
    
    # Progress bar
    progress = used / FREE_USAGE_LIMIT
    st.sidebar.progress(progress)
    
    # Usage stats
    st.sidebar.markdown(f"""
    <div style='background: white; padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <p style='margin: 0; font-size: 14px;'><strong>Analyses Used:</strong> {used}/{FREE_USAGE_LIMIT}</p>
        <p style='margin: 5px 0 0 0; font-size: 14px; color: {"#9370DB" if remaining > 0 else "#ff4444"};'>
            <strong>Remaining:</strong> {remaining}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Warning when low
    if remaining == 1:
        st.sidebar.warning("⚠️ Only 1 analysis remaining!")
    elif remaining == 0:
        st.sidebar.error("❌ Free limit reached!")
        st.sidebar.markdown("""
        <div style='background: linear-gradient(135deg, #9370DB 0%, #BA55D3 100%); 
                    padding: 20px; border-radius: 15px; margin: 10px 0; text-align: center;'>
            <h4 style='color: white; margin: 0 0 10px 0;'>🚀 Upgrade to Premium</h4>
            <p style='color: white; font-size: 14px; margin: 0;'>
                Get unlimited analyses, advanced features, and priority support!
            </p>
        </div>
        """, unsafe_allow_html=True)

def show_upgrade_message():
    """Show upgrade message when limit is reached"""
    st.error("### ❌ Free Usage Limit Reached")
    st.markdown("""
    <div style='background: white; padding: 30px; border-radius: 20px; margin: 20px 0; 
                box-shadow: 0 8px 25px rgba(147, 112, 219, 0.15); border: 2px solid #E6E6FA; text-align: center;'>
        <h2 style='color: #9370DB; margin-top: 0;'>🚀 Upgrade to Premium</h2>
        <p style='color: #4A4A4A; font-size: 18px;'>
            You've used all <strong>5 free analyses</strong>. Upgrade to get:
        </p>
        <ul style='text-align: left; color: #4A4A4A; font-size: 16px; max-width: 500px; margin: 20px auto;'>
            <li>✅ <strong>Unlimited</strong> stock analyses</li>
            <li>✅ <strong>Advanced</strong> portfolio tracking</li>
            <li>✅ <strong>Custom</strong> alerts & notifications</li>
            <li>✅ <strong>Priority</strong> AI processing</li>
            <li>✅ <strong>Export</strong> to Excel & PDF</li>
            <li>✅ <strong>Historical</strong> data analysis</li>
        </ul>
        <p style='color: #9370DB; font-size: 24px; font-weight: bold; margin: 20px 0;'>
            ₹999/month or ₹9,999/year
        </p>
        <p style='color: #666; font-size: 14px; margin: 10px 0;'>
            <em>Contact us to upgrade or request a demo</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
