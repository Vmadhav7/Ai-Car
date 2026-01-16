"""
Streamlit Dashboard for AI-Powered Car Intelligence System

A simple, fast frontend to explore car data, predictions, and recommendations.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data.data_loader import load_cars_data
from data.data_cleaner import clean_dataset, remove_invalid_rows
from data.feature_engineer import create_all_features
from models.price_predictor import PricePredictor
from models.classifier import PerformanceClassifier
from models.recommender import CarRecommender

import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Car Intelligence System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background: #0e1117;
        color: #fafafa;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #1a1a2e;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] .stMetric {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #4ecdc4;
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #a0aec0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1e1e2e;
        padding: 10px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        font-weight: 600;
        background: transparent;
        border-radius: 8px;
        color: #a0aec0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Card styling */
    .metric-card {
        background: #1e1e2e;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2d2d44;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    
    .metric-card h3, .metric-card h4, .metric-card p {
        color: #ffffff !important;
    }
    
    .metric-card h4 {
        color: #4ecdc4 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* All markdown text */
    .stMarkdown, .stMarkdown p, .stMarkdown li {
        color: #e0e0e0;
    }
    
    /* Labels */
    .stSelectbox label, .stSlider label, .stCheckbox label {
        color: #e0e0e0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #ffffff;
        background: #1e1e2e;
    }
    
    .streamlit-expanderContent {
        background: #1e1e2e;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Caption text */
    .stCaption {
        color: #888 !important;
    }
    
    /* Divider */
    hr {
        border-color: #2d2d44;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_prepare_data():
    """Load and prepare the car dataset."""
    df = load_cars_data()
    df_cleaned = clean_dataset(df)
    df_valid = remove_invalid_rows(df_cleaned)
    df_features = create_all_features(df_valid)
    return df_features


@st.cache_resource
def train_models(df):
    """Train the ML models."""
    feature_cols = ['HorsePower_Numeric', 'Speed_KMH', 'Acceleration_Sec',
                    'Torque_Nm', 'CC_Numeric', 'Seats_Numeric']
    
    model_df = df.dropna(subset=feature_cols + ['Price_USD'])
    X = model_df[feature_cols]
    y = model_df['Price_USD']
    
    # Price predictor
    price_predictor = PricePredictor(use_xgboost=False)
    price_predictor.train(X, y, validate=False)
    
    # Classifier
    class_df = model_df[model_df['Performance_Category'] != 'Unknown']
    classifier = PerformanceClassifier(use_xgboost=False)
    classifier.train(class_df[feature_cols], class_df['Performance_Category'], validate=False)
    
    # Recommender
    recommender = CarRecommender()
    recommender.fit(df)
    
    return price_predictor, classifier, recommender, feature_cols


def create_chart_style():
    """Apply consistent dark theme chart styling."""
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#0e1117'
    plt.rcParams['axes.facecolor'] = '#1e1e2e'
    plt.rcParams['axes.edgecolor'] = '#3d3d5c'
    plt.rcParams['axes.labelcolor'] = '#e0e0e0'
    plt.rcParams['text.color'] = '#e0e0e0'
    plt.rcParams['xtick.color'] = '#a0a0a0'
    plt.rcParams['ytick.color'] = '#a0a0a0'
    plt.rcParams['grid.color'] = '#2d2d44'
    plt.rcParams['legend.facecolor'] = '#1e1e2e'
    plt.rcParams['legend.edgecolor'] = '#3d3d5c'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.titleweight'] = 'bold'


def main():
    create_chart_style()
    
    # Header
    st.markdown('<h1 class="main-header">Car Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore data, predict prices, classify performance, and discover similar vehicles</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        try:
            df = load_and_prepare_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    # Train models
    with st.spinner("Preparing models..."):
        try:
            price_predictor, classifier, recommender, feature_cols = train_models(df)
        except Exception as e:
            st.error(f"Error training models: {e}")
            return
    
    # Sidebar - Quick Stats
    with st.sidebar:
        st.markdown("### Dataset Overview")
        st.metric("Total Cars", f"{len(df):,}")
        st.metric("Manufacturers", df['Company_Clean'].nunique())
        st.metric("Average Price", f"${df['Price_USD'].mean():,.0f}")
        st.metric("Average HP", f"{df['HorsePower_Numeric'].mean():.0f}")
        
        st.markdown("---")
        
        st.markdown("### Performance Breakdown")
        category_counts = df['Performance_Category'].value_counts()
        for cat, count in category_counts.items():
            pct = count / len(df) * 100
            st.write(f"**{cat}**")
            st.progress(pct / 100)
            st.caption(f"{count} cars ({pct:.1f}%)")
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Data Explorer", "Price Prediction", "Performance Analysis", "Similar Cars"])
    
    # Tab 1: Data Explorer
    with tab1:
        st.markdown('<p class="section-header">Explore the Dataset</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Price Distribution**")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            prices = df['Price_USD'].dropna()
            upper = prices.quantile(0.95)
            filtered = prices[prices <= upper]
            ax1.hist(filtered, bins=40, color='#667eea', edgecolor='white', alpha=0.85)
            ax1.set_xlabel('Price (USD)', fontsize=11)
            ax1.set_ylabel('Frequency', fontsize=11)
            ax1.ticklabel_format(style='plain', axis='x')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            st.pyplot(fig1)
            plt.close()
        
        with col2:
            st.markdown("**Average Price by Top Brands**")
            top_brands = df.groupby('Company_Clean')['Price_USD'].agg(['mean', 'count'])
            top_brands = top_brands[top_brands['count'] >= 10].nlargest(10, 'mean')
            
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_brands)))
            bars = ax2.barh(top_brands.index, top_brands['mean'], color=colors)
            ax2.set_xlabel('Average Price (USD)', fontsize=11)
            ax2.ticklabel_format(style='plain', axis='x')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            st.pyplot(fig2)
            plt.close()
        
        # Performance scatter
        st.markdown("**Performance vs Price Analysis**")
        col3, col4 = st.columns([3, 1])
        
        with col3:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            scatter_df = df.dropna(subset=['Acceleration_Sec', 'Price_USD'])
            scatter_df = scatter_df[scatter_df['Price_USD'] <= scatter_df['Price_USD'].quantile(0.95)]
            
            colors = {'High-Performance': '#e74c3c', 'Mid-Range': '#3498db', 'Economy': '#27ae60', 'Unknown': '#95a5a6'}
            for cat in ['High-Performance', 'Mid-Range', 'Economy']:
                if cat in scatter_df['Performance_Category'].values:
                    subset = scatter_df[scatter_df['Performance_Category'] == cat]
                    ax3.scatter(subset['Acceleration_Sec'], subset['Price_USD'], 
                               label=cat, alpha=0.6, s=60, c=colors.get(cat, '#95a5a6'), edgecolors='white', linewidth=0.5)
            
            ax3.set_xlabel('0-100 km/h (seconds)', fontsize=11)
            ax3.set_ylabel('Price (USD)', fontsize=11)
            ax3.legend(frameon=True, fancybox=True, shadow=True)
            ax3.ticklabel_format(style='plain', axis='y')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            st.pyplot(fig3)
            plt.close()
        
        with col4:
            st.markdown("**Category Thresholds**")
            st.markdown("""
            - **High-Performance**: < 4.5 sec
            - **Mid-Range**: 4.5 - 8.0 sec
            - **Economy**: > 8.0 sec
            """)
        
        # Data table
        st.markdown("**Browse Cars**")
        display_cols = ['Company_Clean', 'Car_Name_Clean', 'Price_USD', 'HorsePower_Numeric', 
                       'Speed_KMH', 'Acceleration_Sec', 'Fuel_Type_Std', 'Performance_Category']
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            selected_brand = st.selectbox("Filter by Brand", ['All'] + sorted(df['Company_Clean'].unique().tolist()))
        with col_f2:
            selected_category = st.selectbox("Filter by Category", ['All'] + df['Performance_Category'].unique().tolist())
        with col_f3:
            price_range = st.slider("Price Range (USD)", 0, int(df['Price_USD'].max()), (0, 500000))
        
        filtered_df = df.copy()
        if selected_brand != 'All':
            filtered_df = filtered_df[filtered_df['Company_Clean'] == selected_brand]
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['Performance_Category'] == selected_category]
        filtered_df = filtered_df[(filtered_df['Price_USD'] >= price_range[0]) & (filtered_df['Price_USD'] <= price_range[1])]
        
        st.dataframe(
            filtered_df[display_cols].head(50).style.format({
                'Price_USD': '${:,.0f}',
                'HorsePower_Numeric': '{:.0f} HP',
                'Speed_KMH': '{:.0f} km/h',
                'Acceleration_Sec': '{:.1f}s'
            }),
            use_container_width=True,
            height=400
        )
    
    # Tab 2: Price Prediction
    with tab2:
        st.markdown('<p class="section-header">Predict Car Price</p>', unsafe_allow_html=True)
        st.write("Enter car specifications to get an estimated market price.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Power & Performance**")
            hp = st.slider("Horsepower (HP)", 50, 1500, 300)
            torque = st.slider("Torque (Nm)", 50, 1500, 400)
        
        with col2:
            st.markdown("**Speed & Acceleration**")
            speed = st.slider("Top Speed (km/h)", 100, 400, 250)
            acceleration = st.slider("0-100 km/h (seconds)", 2.0, 20.0, 6.0, 0.1)
        
        with col3:
            st.markdown("**Engine & Capacity**")
            cc = st.slider("Engine Displacement (CC)", 500, 8000, 3000)
            seats = st.selectbox("Seating Capacity", [2, 4, 5, 7], index=2)
        
        st.markdown("---")
        
        if st.button("Predict Price", type="primary"):
            input_data = pd.DataFrame({
                'HorsePower_Numeric': [hp],
                'Speed_KMH': [speed],
                'Acceleration_Sec': [acceleration],
                'Torque_Nm': [torque],
                'CC_Numeric': [cc],
                'Seats_Numeric': [seats]
            })
            
            try:
                predicted_price = price_predictor.predict(input_data)[0]
                category = classifier.predict(input_data)[0]
                
                col_r1, col_r2 = st.columns(2)
                
                with col_r1:
                    st.success(f"### Estimated Price: ${predicted_price:,.0f}")
                
                with col_r2:
                    category_styles = {
                        'High-Performance': ('error', 'High-Performance Vehicle'),
                        'Mid-Range': ('info', 'Mid-Range Vehicle'),
                        'Economy': ('success', 'Economy Vehicle')
                    }
                    style, label = category_styles.get(category, ('info', category))
                    if style == 'error':
                        st.error(f"### {label}")
                    elif style == 'success':
                        st.success(f"### {label}")
                    else:
                        st.info(f"### {label}")
                
                # Feature importance
                st.markdown("**Key Price Factors**")
                importance = price_predictor.get_feature_importance()
                
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance)))
                bars = ax.barh(importance['feature'], importance['importance'], color=colors)
                ax.set_xlabel('Importance Score', fontsize=11)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
    
    # Tab 3: Performance Analysis
    with tab3:
        st.markdown('<p class="section-header">Performance Category Analysis</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distribution by Category**")
            category_counts = df['Performance_Category'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#e74c3c', '#3498db', '#27ae60', '#95a5a6']
            wedges, texts, autotexts = ax.pie(
                category_counts, 
                labels=category_counts.index, 
                autopct='%1.1f%%', 
                colors=colors[:len(category_counts)], 
                startangle=90,
                explode=[0.02] * len(category_counts)
            )
            for autotext in autotexts:
                autotext.set_fontweight('bold')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("**Average Statistics by Category**")
            stats = df.groupby('Performance_Category').agg({
                'Price_USD': 'mean',
                'HorsePower_Numeric': 'mean',
                'Speed_KMH': 'mean',
                'Acceleration_Sec': 'mean'
            }).round(0)
            stats.columns = ['Avg Price ($)', 'Avg HP', 'Avg Speed (km/h)', 'Avg 0-100 (sec)']
            
            st.dataframe(
                stats.style.format({
                    'Avg Price ($)': '${:,.0f}',
                    'Avg HP': '{:.0f}',
                    'Avg Speed (km/h)': '{:.0f}',
                    'Avg 0-100 (sec)': '{:.1f}'
                }),
                use_container_width=True
            )
        
        # Top cars by category
        st.markdown("**Top Performers by Category**")
        for category in ['High-Performance', 'Mid-Range', 'Economy']:
            with st.expander(f"{category} - Top 5 Fastest"):
                cat_df = df[df['Performance_Category'] == category].nsmallest(5, 'Acceleration_Sec')
                st.dataframe(
                    cat_df[['Company_Clean', 'Car_Name_Clean', 'Acceleration_Sec', 'HorsePower_Numeric', 'Price_USD']].style.format({
                        'Acceleration_Sec': '{:.1f}s',
                        'HorsePower_Numeric': '{:.0f} HP',
                        'Price_USD': '${:,.0f}'
                    }),
                    use_container_width=True
                )
    
    # Tab 4: Similar Cars
    with tab4:
        st.markdown('<p class="section-header">Find Similar Cars</p>', unsafe_allow_html=True)
        st.write("Select a car to discover similar vehicles based on specifications.")
        
        # Select a car
        car_options = df['Company_Clean'] + " " + df['Car_Name_Clean']
        
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_car = st.selectbox("Select a car:", car_options.tolist())
        with col2:
            n_recs = st.slider("Results", 3, 10, 5)
        
        exclude_brand = st.checkbox("Exclude same manufacturer")
        
        if st.button("Find Similar Cars", type="primary"):
            try:
                idx = car_options.tolist().index(selected_car)
                recommendations = recommender.get_similar_cars(idx, n_recommendations=n_recs, 
                                                               exclude_same_brand=exclude_brand)
                
                target_car = df.iloc[idx]
                
                st.markdown("---")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Selected Vehicle**")
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0; color:#1a1a2e;">{target_car['Company_Clean']}</h3>
                        <h4 style="margin:0; color:#667eea;">{target_car['Car_Name_Clean']}</h4>
                        <br>
                        <p><strong>Price:</strong> ${target_car['Price_USD']:,.0f}</p>
                        <p><strong>Power:</strong> {target_car['HorsePower_Numeric']:.0f} HP</p>
                        <p><strong>0-100:</strong> {target_car['Acceleration_Sec']:.1f} seconds</p>
                        <p><strong>Category:</strong> {target_car['Performance_Category']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**Similar Vehicles**")
                    for i, (_, car) in enumerate(recommendations.iterrows(), 1):
                        similarity_pct = car['similarity_score'] * 100
                        st.markdown(f"""
                        **{i}. {car['Company_Clean']} {car['Car_Name_Clean']}**  
                        Price: ${car['Price_USD']:,.0f} | Power: {car['HorsePower_Numeric']:.0f} HP | Match: {similarity_pct:.0f}%
                        """)
                        st.progress(similarity_pct / 100)
                
            except Exception as e:
                st.error(f"Error finding similar cars: {e}")


if __name__ == "__main__":
    main()
