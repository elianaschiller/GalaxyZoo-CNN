callbacks = [
    keras.callbacks.ModelCheckpoint(  # outputs training/validation accuracy and loss for each epoch trained
        filepath='CNN_Model_Checkpoints.keras',
        save_best_only=True,  # saves model file of best epoch
        monitor='val_loss'
    ),
    keras.callbacks.EarlyStopping(  # stops training when 10 epochs run with no improvement of validation loss
        monitor='val_loss',  
        patience=10,  
        restore_best_weights=True
    )
]

# train model with 100 epochs
history = model.fit(
    train_generator,
    epochs=100, 
    validation_data=val_generator,
    callbacks=callbacks
)
