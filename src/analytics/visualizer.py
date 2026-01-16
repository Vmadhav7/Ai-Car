"""
Visualization module for the Car Intelligence System.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')


def plot_price_distribution(df: pd.DataFrame,
                            price_col: str = 'Price_USD',
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot price distribution with histogram and box plot.
    
    Args:
        df: DataFrame with price data
        price_col: Name of the price column
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Remove outliers for better visualization (prices > 99th percentile)
    prices = df[price_col].dropna()
    upper_limit = prices.quantile(0.99)
    filtered_prices = prices[prices <= upper_limit]
    
    # Histogram
    axes[0].hist(filtered_prices, bins=50, edgecolor='white', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Price (USD)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Car Price Distribution', fontsize=14, fontweight='bold')
    axes[0].ticklabel_format(style='plain', axis='x')
    
    # Box plot
    sns.boxplot(x=filtered_prices, ax=axes[1], color='steelblue')
    axes[1].set_xlabel('Price (USD)', fontsize=12)
    axes[1].set_title('Price Box Plot', fontsize=14, fontweight='bold')
    axes[1].ticklabel_format(style='plain', axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_heatmap(df: pd.DataFrame,
                              columns: Optional[List[str]] = None,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot correlation heatmap for numerical features.
    
    Args:
        df: DataFrame with numerical data
        columns: Specific columns to include
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, square=True,
                linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_brand_price_boxplot(df: pd.DataFrame,
                              company_col: str = 'Company_Clean',
                              price_col: str = 'Price_USD',
                              top_n: int = 15,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Plot price distribution by brand (top N brands by count).
    
    Args:
        df: DataFrame with car data
        company_col: Company column name
        price_col: Price column name
        top_n: Number of top brands to show
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    # Get top N brands by count
    top_brands = df[company_col].value_counts().head(top_n).index.tolist()
    filtered_df = df[df[company_col].isin(top_brands)]
    
    # Order by median price
    brand_order = filtered_df.groupby(company_col)[price_col].median().sort_values(ascending=False).index
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.boxplot(data=filtered_df, x=company_col, y=price_col, 
                order=brand_order, palette='viridis', ax=ax)
    
    ax.set_xlabel('Brand', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.set_title(f'Price Distribution by Top {top_n} Brands', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_fuel_type_distribution(df: pd.DataFrame,
                                 fuel_col: str = 'Fuel_Type_Std',
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot fuel type distribution as bar chart and pie chart.
    
    Args:
        df: DataFrame with car data
        fuel_col: Fuel type column name
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    fuel_counts = df[fuel_col].value_counts()
    
    # Bar chart
    fuel_counts.plot(kind='bar', ax=axes[0], color=sns.color_palette('husl', len(fuel_counts)))
    axes[0].set_xlabel('Fuel Type', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Cars by Fuel Type', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Pie chart
    axes[1].pie(fuel_counts, labels=fuel_counts.index, autopct='%1.1f%%',
                colors=sns.color_palette('husl', len(fuel_counts)))
    axes[1].set_title('Fuel Type Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_performance_scatter(df: pd.DataFrame,
                              x_col: str = 'Acceleration_Sec',
                              y_col: str = 'Speed_KMH',
                              color_col: str = 'Price_USD',
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot scatter plot of performance metrics.
    
    Args:
        df: DataFrame with car data
        x_col: X-axis column
        y_col: Y-axis column
        color_col: Column for color encoding
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    # Remove missing values
    plot_df = df[[x_col, y_col, color_col]].dropna()
    
    # Limit color range for better visualization
    color_max = plot_df[color_col].quantile(0.95)
    plot_df = plot_df[plot_df[color_col] <= color_max]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(plot_df[x_col], plot_df[y_col], 
                         c=plot_df[color_col], cmap='viridis',
                         alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Price (USD)', fontsize=12)
    
    ax.set_xlabel('0-100 km/h (seconds)', fontsize=12)
    ax.set_ylabel('Top Speed (km/h)', fontsize=12)
    ax.set_title('Performance: Acceleration vs Top Speed (colored by Price)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_horsepower_vs_price(df: pd.DataFrame,
                              hp_col: str = 'HorsePower_Numeric',
                              price_col: str = 'Price_USD',
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot horsepower vs price with regression line.
    
    Args:
        df: DataFrame with car data
        hp_col: Horsepower column name
        price_col: Price column name
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    # Remove missing values and outliers
    plot_df = df[[hp_col, price_col]].dropna()
    price_upper = plot_df[price_col].quantile(0.95)
    plot_df = plot_df[plot_df[price_col] <= price_upper]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.regplot(data=plot_df, x=hp_col, y=price_col, 
                scatter_kws={'alpha': 0.4, 's': 30},
                line_kws={'color': 'red', 'linewidth': 2},
                ax=ax)
    
    ax.set_xlabel('Horsepower (HP)', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.set_title('Horsepower vs Price', fontsize=14, fontweight='bold')
    ax.ticklabel_format(style='plain', axis='y')
    
    # Add correlation text
    correlation = plot_df[hp_col].corr(plot_df[price_col])
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_performance_category_distribution(df: pd.DataFrame,
                                            category_col: str = 'Performance_Category',
                                            price_col: str = 'Price_USD',
                                            save_path: Optional[str] = None,
                                            figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    Plot performance category distribution and price comparison.
    
    Args:
        df: DataFrame with car data
        category_col: Performance category column
        price_col: Price column
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    category_order = ['High-Performance', 'Mid-Range', 'Economy']
    
    # Count plot
    category_counts = df[category_col].value_counts().reindex(category_order).fillna(0)
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    axes[0].bar(category_counts.index, category_counts.values, color=colors)
    axes[0].set_xlabel('Performance Category', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Cars by Performance Category', fontsize=14, fontweight='bold')
    
    # Price boxplot by category
    plot_df = df[df[category_col].isin(category_order)]
    sns.boxplot(data=plot_df, x=category_col, y=price_col,
                order=category_order, palette=colors, ax=axes[1])
    axes[1].set_xlabel('Performance Category', fontsize=12)
    axes[1].set_ylabel('Price (USD)', fontsize=12)
    axes[1].set_title('Price by Performance Category', fontsize=14, fontweight='bold')
    axes[1].ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame,
                            title: str = 'Feature Importance',
                            top_n: int = 10,
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot feature importance as horizontal bar chart.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        title: Plot title
        top_n: Number of top features to show
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    # Get top N features
    plot_df = importance_df.head(top_n).sort_values('importance')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(plot_df['feature'], plot_df['importance'], color='steelblue')
    
    # Add value labels
    for bar, val in zip(bars, plot_df['importance']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=10)
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(cm: np.ndarray,
                          labels: List[str],
                          title: str = 'Confusion Matrix',
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_all_visualizations(df: pd.DataFrame,
                               output_dir: str) -> List[str]:
    """
    Generate and save all standard visualizations.
    
    Args:
        df: Cleaned DataFrame with all features
        output_dir: Directory to save figures
        
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Generate each visualization
    visualizations = [
        ('price_distribution.png', lambda: plot_price_distribution(df)),
        ('correlation_heatmap.png', lambda: plot_correlation_heatmap(df)),
        ('fuel_type_distribution.png', lambda: plot_fuel_type_distribution(df)),
    ]
    
    if 'Company_Clean' in df.columns and 'Price_USD' in df.columns:
        visualizations.append(
            ('brand_price_boxplot.png', lambda: plot_brand_price_boxplot(df))
        )
    
    if 'Acceleration_Sec' in df.columns and 'Speed_KMH' in df.columns:
        visualizations.append(
            ('performance_scatter.png', lambda: plot_performance_scatter(df))
        )
    
    if 'HorsePower_Numeric' in df.columns and 'Price_USD' in df.columns:
        visualizations.append(
            ('horsepower_vs_price.png', lambda: plot_horsepower_vs_price(df))
        )
    
    if 'Performance_Category' in df.columns:
        visualizations.append(
            ('performance_category.png', lambda: plot_performance_category_distribution(df))
        )
    
    for filename, plot_func in visualizations:
        try:
            filepath = output_path / filename
            fig = plot_func()
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(str(filepath))
        except Exception as e:
            print(f"Error creating {filename}: {e}")
    
    return saved_files
