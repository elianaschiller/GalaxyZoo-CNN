# using sklearn to split sample into 70% train, 15% validation, 15% test
train_df[["Elliptical", "Disk"]] = train_df[["Elliptical", "Disk"]].astype(float)
val_df[["Elliptical", "Disk"]] = val_df[["Elliptical", "Disk"]].astype(float)

# image augmentation to reduce overfitting
datagen7=ImageDataGenerator(rescale=1./255.,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest'
                           )

# generating batches of train, validation, and test data
train_generator7=datagen7.flow_from_dataframe(
    dataframe=train_df,
    directory=data_path,
    x_col="GalaxyID",
    y_col=["Elliptical", "Disk"],
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="other",
    target_size=(64,64))

val_datagen7 = ImageDataGenerator(rescale=1./255)

val_generator7=val_datagen7.flow_from_dataframe(
    dataframe=val_df,
    directory=data_path,
    x_col="GalaxyID",
    y_col=["Elliptical", "Disk"],
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="other",
    target_size=(64,64))

test_datagen7 = ImageDataGenerator(rescale=1./255)

test_generator7 = test_datagen7.flow_from_dataframe(
    dataframe=test_df,
    directory=data_path,
    x_col="GalaxyID",
    y_col=["Elliptical", "Disk"],  
    batch_size=3,
    seed=42,
    shuffle=False,  
    class_mode="other",
    target_size=(64, 64)
)
