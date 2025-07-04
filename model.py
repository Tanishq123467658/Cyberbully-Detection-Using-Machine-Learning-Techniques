import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, learning_curve
from imblearn.over_sampling import SMOTE
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

class CyberbullyingClassifier:
    def __init__(self, model_type='logistic', class_weight='balanced', random_state=42):
        """
        Initialize classifier
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('logistic', 'rf', 'svm', 'gb')
        class_weight : str or dict
            Class weights for handling imbalanced data
        random_state : int
            Random seed for reproducibility
        """
        self.model_type = model_type
        self.class_weight = class_weight
        self.random_state = random_state
        
        if model_type == 'logistic':
            self.model = LogisticRegression(
                class_weight=class_weight,
                random_state=random_state,
                max_iter=1000,
                C=1.0,
                solver='liblinear'
            )
        elif model_type == 'rf':
            self.model = RandomForestClassifier(
                class_weight=class_weight,
                random_state=random_state,
                n_estimators=100,
                max_depth=None
            )
        elif model_type == 'svm':
            self.model = SVC(
                class_weight=class_weight,
                random_state=random_state,
                probability=True,
                kernel='rbf',
                C=1.0
            )
        elif model_type == 'gb':
            self.model = GradientBoostingClassifier(
                random_state=random_state,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, use_smote=False):
        """
        Train the model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation labels
        use_smote : bool
            Whether to use SMOTE for oversampling
        """
        # Apply SMOTE if requested
        if use_smote:
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Class distribution: {np.bincount(y_train)}")
        
        # Train the model
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Predict class labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val, param_grid=None):
        """
        Tune hyperparameters using grid search
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        X_val : array-like
            Validation features
        y_val : array-like
            Validation labels
        param_grid : dict, optional
            Parameter grid for grid search
        
        Returns:
        --------
        dict
            Best parameters
        """
        if param_grid is None:
            if self.model_type == 'logistic':
                param_grid = {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'saga'],
                    'penalty': ['l1', 'l2']
                }
            elif self.model_type == 'rf':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto', 0.1, 0.01]
                }
            elif self.model_type == 'gb':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
        
        # Combine train and validation sets for cross-validation
        X_combined = np.vstack([X_train.toarray(), X_val.toarray()]) if hasattr(X_train, 'toarray') else np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_combined, y_combined)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def plot_learning_curve(self, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5), save_path='results/learning_curve.png'):
        """
        Plot learning curve for the model
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Labels
        cv : int
            Number of cross-validation folds
        train_sizes : array-like
            Training set sizes to plot
        save_path : str
            Path to save the plot
        """
        # Reduce memory usage by using fewer samples and fewer cross-validation folds
        # Sample the data if it's too large (more than 10,000 samples)
        if X.shape[0] > 10000:
            sample_size = 10000
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sampled = X[indices] if not hasattr(X, 'toarray') else X[indices].toarray()
            y_sampled = y[indices]
        else:
            X_sampled = X if not hasattr(X, 'toarray') else X.toarray()
            y_sampled = y
        
        # Use fewer CV folds and run with only 1 job to reduce memory usage
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X_sampled, y_sampled, cv=3, n_jobs=1, 
            train_sizes=train_sizes, scoring='f1'
        )
        
        # Rest of the function remains the same
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Learning Curve ({self.model_type})')
        plt.xlabel('Training examples')
        plt.ylabel('F1 Score')
        plt.grid()
        
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color='blue')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color='orange')
        plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='orange', label='Cross-validation score')
        
        plt.legend(loc='best')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    
    def save(self, file_path):
        """Save the model"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, file_path):
        """Load the model"""
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)

def plot_confusion_matrix(cm, classes=['Not Cyberbullying', 'Cyberbullying'], title='Confusion Matrix'):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_test, y_prob, title='ROC Curve'):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/roc_curve.png')
    plt.close()

def train_and_evaluate_model(data, model_type='logistic', use_smote=False, tune_params=False):
    """
    Train and evaluate a model
    
    Parameters:
    -----------
    data : dict
        Dictionary containing train, validation, and test data
    model_type : str
        Type of model to use
    use_smote : bool
        Whether to use SMOTE for oversampling
    tune_params : bool
        Whether to tune hyperparameters
    
    Returns:
    --------
    tuple
        Trained model and evaluation results
    """
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize classifier
    classifier = CyberbullyingClassifier(model_type=model_type)
    
    # Tune hyperparameters if requested
    if tune_params:
        # Sample data for hyperparameter tuning to reduce memory usage
        if hasattr(data['X_train'], 'toarray'):
            X_train_sample = data['X_train'][:5000].toarray()
            X_val_sample = data['X_val'][:1000].toarray()
        else:
            X_train_sample = data['X_train'][:5000]
            X_val_sample = data['X_val'][:1000]
        
        y_train_sample = data['y_train'][:5000]
        y_val_sample = data['y_val'][:1000]
        
        classifier.tune_hyperparameters(
            X_train_sample, y_train_sample,
            X_val_sample, y_val_sample
        )
    
    # Train the model
    classifier.fit(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        use_smote=use_smote
    )
    
    # Plot learning curve with sampled data
    # We'll create a smaller combined dataset to reduce memory usage
    sample_size_train = min(5000, data['X_train'].shape[0])
    sample_size_val = min(1000, data['X_val'].shape[0])
    
    if hasattr(data['X_train'], 'toarray'):
        X_train_sample = data['X_train'][:sample_size_train].toarray()
        X_val_sample = data['X_val'][:sample_size_val].toarray()
        X_combined = np.vstack([X_train_sample, X_val_sample])
    else:
        X_train_sample = data['X_train'][:sample_size_train]
        X_val_sample = data['X_val'][:sample_size_val]
        X_combined = np.vstack([X_train_sample, X_val_sample])
    
    y_combined = np.concatenate([
        data['y_train'][:sample_size_train], 
        data['y_val'][:sample_size_val]
    ])
    
    classifier.plot_learning_curve(X_combined, y_combined)
    
    # Evaluate on test set
    results = classifier.evaluate(data['X_test'], data['y_test'])
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(classification_report(data['y_test'], results['y_pred']))
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'])
    
    # Plot ROC curve
    plot_roc_curve(data['y_test'], results['y_prob'])
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    classifier.save(f'models/{model_type}_model.pkl')
    
    # If the model supports feature importance, plot it
    if hasattr(classifier.model, 'coef_') or hasattr(classifier.model, 'feature_importances_'):
        from visualization import plot_feature_importance
        
        if hasattr(data['feature_extractor'], 'vectorizer'):
            feature_names = data['feature_extractor'].vectorizer.get_feature_names_out()
            plot_feature_importance(classifier.model, feature_names)
    
    return classifier, results

if __name__ == "__main__":
    from preprocess import load_and_preprocess_data
    from feature_extraction import prepare_data
    
    # Load and preprocess data
    df = load_and_preprocess_data("cyberbullying_tweets.csv")
    
    # Prepare data
    data = prepare_data(df, feature_method='tfidf')
    
    # Train and evaluate model
    classifier, results = train_and_evaluate_model(data, model_type='logistic', use_smote=True)