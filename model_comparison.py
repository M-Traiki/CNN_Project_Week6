#!/usr/bin/env python
# Script to compare models and measure prediction speed

import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

def main():
    print("Starting Model Comparison Analysis...")
    
    # Load the CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Define all models
    all_models = [
        ("initial_model.keras", "Initial CNN Baseline", True, "categorical_crossentropy", False),  # needs grayscale, categorical_crossentropy, no unprocessed
        ("best_model_RMS_with_drop.keras", "Improved CNN RMSprop with Dropout", True, "categorical_crossentropy", False),  # needs grayscale, categorical_crossentropy, no unprocessed
        ("Fine_Tuned_Eff_RMS_unfrozen_with_resize.keras", "Fine-tuned EfficientNet RMS", False, "sparse_categorical_crossentropy", True),  # no grayscale, sparse_categorical_crossentropy, unprocessed data
        ("EffNET0_val_accuracy_with_resize.keras", "EfficientNet Baseline", False, "sparse_categorical_crossentropy", True)  # no grayscale, sparse_categorical_crossentropy, unprocessed data
    ]
    results = []
    
    for model_path, model_name, needs_grayscale, loss_type, use_unprocessed in all_models:
        try:
            print(f"\nLoading model: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"Model file {model_path} does not exist. Skipping.")
                continue
            
            # Enable unsafe deserialization for lambda layers
            keras.config.enable_unsafe_deserialization()
            import tensorflow as tf
            custom_objects = {"tf": tf}
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
            
            print(f"Evaluating {model_name}...")
            print(f"  Expected input: {'grayscale' if needs_grayscale else 'RGB'}, loss type: {loss_type}")
            
            # Prepare data based on model requirements
            if use_unprocessed:
                # For models with "with_resize", use RGB format without additional processing
                # These models have their own internal preprocessing
                x_test_processed = x_test.astype('float32')  # Just convert to float32, no division
                print(f"  Using unprocessed RGB data shape: {x_test_processed.shape}")
            elif needs_grayscale:
                # Convert RGB to grayscale and normalize for grayscale models
                x_test_processed = np.dot(x_test[...,:3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale conversion
                x_test_processed = np.expand_dims(x_test_processed, -1)  # Add channel dimension back
                x_test_processed = x_test_processed.astype('float32') / 255.0  # Normalize
                print(f"  Processed grayscale data shape: {x_test_processed.shape}")
            else:
                # For other RGB models, normalize 
                x_test_processed = x_test.astype('float32') / 255.0
                print(f"  Using normalized RGB data shape: {x_test_processed.shape}")
            
            # Prepare labels based on loss type
            if loss_type == "sparse_categorical_crossentropy":
                # Integer labels (shape: (n, 1) -> flatten to (n,))
                y_test_processed = y_test.flatten()  # Convert to integer format
            else:  # categorical_crossentropy
                # One-hot encoded labels
                y_test_processed = keras.utils.to_categorical(y_test, 10)
            
            print(f"  Label format: {'integer' if loss_type == 'sparse_categorical_crossentropy' else 'one-hot'}")
            
            # Measure evaluation time and metrics
            start_time = time.time()
            test_loss, test_accuracy = model.evaluate(x_test_processed, y_test_processed, verbose=0)
            evaluation_time = time.time() - start_time
            
            # Get predictions for the full test set to calculate additional metrics
            predictions = model.predict(x_test_processed, verbose=0)
            
            # Convert predictions to class labels
            predicted_classes = np.argmax(predictions, axis=1)
            
            # For metrics calculation, use the appropriate true labels format
            if loss_type == "sparse_categorical_crossentropy":
                true_classes = y_test.flatten()  # Use original integer labels
            else:  # categorical_crossentropy
                true_classes = np.argmax(y_test_processed, axis=1)  # Convert one-hot to integers
            
            # Calculate precision, recall, and F1-score
            from sklearn.metrics import precision_recall_fscore_support, accuracy_score
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_classes, predicted_classes, average='weighted', zero_division=0
            )
            
            # Recalculate accuracy from predictions to confirm
            calculated_accuracy = accuracy_score(true_classes, predicted_classes)
            
            # Measure prediction speed
            # Use a subset for speed testing to avoid long waits
            test_subset = x_test_processed[:100]  # Use first 100 samples for speed test
            start_time = time.time()
            predictions_subset = model.predict(test_subset, verbose=0)
            prediction_time = time.time() - start_time
            
            avg_prediction_time_per_sample = prediction_time / len(test_subset)
            
            result = {
                'model_name': model_name,
                'test_accuracy': test_accuracy,
                'calculated_accuracy': calculated_accuracy,
                'test_loss': test_loss,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'evaluation_time': evaluation_time,
                'avg_prediction_time_per_sample': avg_prediction_time_per_sample * 1000,  # Convert to milliseconds
            }
            
            results.append(result)
            
            print(f"  Accuracy: {test_accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Loss: {test_loss:.4f}")
            print(f"  Evaluation Time: {evaluation_time:.4f}s")
            print(f"  Avg Prediction Time per Sample: {avg_prediction_time_per_sample*1000:.4f}ms")
            
        except Exception as e:
            print(f"Error with model {model_path}: {str(e)}")
    
    if results:
        print("\n" + "="*120)
        print("COMPREHENSIVE MODEL COMPARISON RESULTS")
        print("="*120)
        print(f"{'Model Name':<45} {'Accuracy':<8} {'Precision':<10} {'Recall':<8} {'F1-Score':<10} {'Loss':<8} {'Eval Time':<10} {'Pred Time/ms':<12}")
        print("-"*120)
        
        for result in results:
            print(f"{result['model_name']:<45} {result['test_accuracy']:<8.4f} {result['precision']:<10.4f} "
                  f"{result['recall']:<8.4f} {result['f1_score']:<10.4f} {result['test_loss']:<8.4f} "
                  f"{result['evaluation_time']:<10.4f} {result['avg_prediction_time_per_sample']:<12.4f}")
        
        # Find the best models for each metric
        best_accuracy = max(results, key=lambda x: x['test_accuracy'])
        best_precision = max(results, key=lambda x: x['precision'])
        best_recall = max(results, key=lambda x: x['recall'])
        best_f1 = max(results, key=lambda x: x['f1_score'])
        fastest_model = min(results, key=lambda x: x['avg_prediction_time_per_sample'])
        
        print("\nSUMMARY:")
        print(f"Best Accuracy: {best_accuracy['model_name']} ({best_accuracy['test_accuracy']:.4f})")
        print(f"Best Precision: {best_precision['model_name']} ({best_precision['precision']:.4f})")
        print(f"Best Recall: {best_recall['model_name']} ({best_recall['recall']:.4f})")
        print(f"Best F1-Score: {best_f1['model_name']} ({best_f1['f1_score']:.4f})")
        print(f"Fastest Model: {fastest_model['model_name']} ({fastest_model['avg_prediction_time_per_sample']:.4f}ms per sample)")
        
        # Create comparison visualization
        model_names = [r['model_name'] for r in results]
        accuracies = [r['test_accuracy'] for r in results]
        prediction_times = [r['avg_prediction_time_per_sample'] for r in results]
        
        # Plot accuracy comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        bars1 = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, max(accuracies) * 1.1)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot prediction speed comparison
        plt.subplot(1, 2, 2)
        bars2 = plt.bar(model_names, prediction_times, color=['gold', 'orange', 'tomato', 'violet'])
        plt.title('Model Prediction Speed Comparison')
        plt.ylabel('Prediction Time per Sample (ms)')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, time_ms in zip(bars2, prediction_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(prediction_times)*0.01, 
                    f'{time_ms:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        print("No models were successfully evaluated.")

if __name__ == "__main__":
    main()