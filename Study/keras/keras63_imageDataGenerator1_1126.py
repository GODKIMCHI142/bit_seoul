from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지에 대한 생성옵션 정하기
# 1. 증폭기능
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True, 
                                   vertical_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rotation_range=5,
                                   zoom_range=1.2,
                                   shear_range=0.7,
                                   fill_mode="nearest")

test_datagen = ImageDataGenerator(rescale=1./255)




# flow 또는 flow_from_directory
# 파일은 flow 폴더는 flow_from_directory
# 실제 데이터가 있는 곳을 알려주고, 이미지를 불러오는 작업.
train_generator = train_datagen.flow_from_directory(
    "./data/data1/train",
    target_size=(150,150),
    batch_size=5, # image를 5장씩
    class_mode="binary"
)

test_generator = test_datagen.flow_from_directory(
    "./data/data1/test",
    target_size=(150,150),
    batch_size=5, # image를 5장씩
    class_mode="binary"
)


model.fit_generator(
    train_generator,
    steps_per_epochs = 100,
    epochs           = 20,
    validation_data  = test_generator,
    validation_steps = 4
)





