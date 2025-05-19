import os
import sys
from PyQt5.QtWidgets import QApplication, QMessageBox

def main():
    # Check if the model exists
    if not os.path.exists('saved_models/xgb_model.pkl') or not os.path.exists('saved_models/encoders.pkl'):
        print("Model files not found. Running save_model.py first...")
        
        # If in interactive terminal, ask user if they want to run save_model.py
        try:
            # Create a temporary application instance for the dialog
            temp_app = QApplication([])
            reply = QMessageBox.question(None, 'Model Not Found',
                                         'The required model files were not found. Would you like to train and save the model now?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            
            if reply == QMessageBox.Yes:
                try:
                    import save_model
                    save_model.save_trained_model()
                except Exception as e:
                    QMessageBox.critical(None, 'Error',
                                         f'Failed to train and save the model: {str(e)}')
                    return 1
            else:
                QMessageBox.information(None, 'Information',
                                        'The application will run, but predictions may not work correctly.')
            
            # Clean up the temporary application
            temp_app = None
        except Exception as e:
            # Non-interactive environment or PyQt not installed
            print(f"GUI dialog failed: {e}. Trying to train model directly.")
            try:
                import save_model
                save_model.save_trained_model()
            except Exception as e:
                print(f"Error: Failed to train and save the model: {e}")
                return 1
    
    # Now run the GUI application
    from obesity_prediction_gui import ObesityPredictionApp
    
    # Start the application
    app = QApplication(sys.argv)
    window = ObesityPredictionApp()
    window.show()
    return app.exec_()

if __name__ == "__main__":
    sys.exit(main()) 