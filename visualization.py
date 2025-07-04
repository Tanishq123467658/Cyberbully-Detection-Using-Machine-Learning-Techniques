import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def plot_class_distribution(df, save_path='results/class_distribution.png'):
    """Plot class distribution"""
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=df, palette='viridis')
    plt.title('Class Distribution')
    plt.xlabel('Class (0: Not Cyberbullying, 1: Cyberbullying)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Not Cyberbullying', 'Cyberbullying'])
    
    # Add count labels
    for i, count in enumerate(df['label'].value_counts().sort_index()):
        plt.text(i, count + 100, str(count), ha='center')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_text_length_distribution(df, save_path='results/text_length_distribution.png'):
    """Plot text length distribution by class"""
    df['text_length'] = df['processed_text'].apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='text_length', hue='label', bins=50, kde=True, palette='viridis')
    plt.title('Text Length Distribution by Class')
    plt.xlabel('Text Length (words)')
    plt.ylabel('Count')
    plt.legend(['Not Cyberbullying', 'Cyberbullying'])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_wordcloud(df, class_label, save_path='results/wordcloud.png'):
    """Generate word cloud for a specific class"""
    text = ' '.join(df[df['label'] == class_label]['processed_text'])
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=200,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud - {'Cyberbullying' if class_label == 1 else 'Not Cyberbullying'}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(model, feature_names, top_n=20, save_path='results/feature_importance.png'):
    """Plot feature importance for a model"""
    if hasattr(model, 'coef_'):
        # For linear models
        importance = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models
        importance = model.feature_importances_
    else:
        print("Model doesn't support feature importance extraction")
        return
    
    # Get top features
    indices = np.argsort(importance)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance[indices]
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_importance, y=top_features, palette='viridis')
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_data_visualization(X, y, method='pca', save_path='results/data_visualization.png'):
    """Visualize data using dimensionality reduction"""
    # Convert sparse matrix to dense if needed
    if hasattr(X, 'toarray'):
        X = X.toarray()
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Reduce dimensions
    X_reduced = reducer.fit_transform(X)
    
    # Create DataFrame for plotting
    df_plot = pd.DataFrame({
        'x': X_reduced[:, 0],
        'y': X_reduced[:, 1],
        'label': y
    })
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='x', y='y', hue='label', data=df_plot, palette='viridis', alpha=0.7)
    plt.title(f'Data Visualization using {method.upper()}')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.legend(['Not Cyberbullying', 'Cyberbullying'])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_learning_curve(train_sizes, train_scores, test_scores, save_path='results/learning_curve.png'):
    """Plot learning curve"""
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.grid()
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='orange')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training Score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='orange', label='Cross-validation Score')
    
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_data(df):
    """Generate all visualizations"""
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Plot class distribution
    plot_class_distribution(df)
    
    # Plot text length distribution
    plot_text_length_distribution(df)
    
    # Generate word clouds
    generate_wordcloud(df, 0, 'results/wordcloud_not_cyberbullying.png')
    generate_wordcloud(df, 1, 'results/wordcloud_cyberbullying.png')
    
    print("Data visualizations generated successfully!")

if __name__ == "__main__":
    from preprocess import load_and_preprocess_data
    
    # Load and preprocess data
    df = load_and_preprocess_data("cyberbullying_tweets.csv")
    
    # Generate visualizations
    visualize_data(df)