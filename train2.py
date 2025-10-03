from sklearn.kernel_ridge import KernelRidge
from misc import load_data, preprocess_data, split_data, train_model, evaluate_model

def main():
    # Load and preprocess data
    df = load_data()
    feature_cols = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    target_col = 'MEDV'
    X, y = preprocess_data(df, feature_cols, target_col)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Initialize and train Kernel Ridge model
    model = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.1)
    trained_model = train_model(model, X_train, y_train)

    # Evaluate model
    mse = evaluate_model(trained_model, X_test, y_test)
    print(f"Average Mean Squared Error on test set: {mse:.4f}")

if __name__ == "__main__":
    main()

