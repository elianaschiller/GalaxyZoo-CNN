test_model = keras.models.load_model('CNN_Model_Checkpoints.keras')  # loads checkpoint file
print(test_model.summary())  # summary of test model
test_loss, test_acc = test_model.evaluate(test_generator)  # evaluate test accuracy
print(f'Test accuracy: {test_acc:.3f}')

probabilities = test_model.predict(test_generator)  #print morphology predictions
print(probabilities)
